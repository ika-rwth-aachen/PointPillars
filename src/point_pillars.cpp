#define _USE_MATH_DEFINES
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>

struct IntPairHash {
  std::size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
    assert(sizeof(std::size_t) >= 8);
    // Shift first integer over to make room for the second integer. The two are
    // then packed side by side.
    return (((uint64_t)p.first) << 32) | ((uint64_t)p.second);
  }
};

struct PillarPoint {
  float x;
  float y;
  float z;
  float intensity;
  float xc;
  float yc;
  float zc;
};

template <class T>
const T& clamp(const T& v, const T& lo, const T& hi) {
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

pybind11::tuple createPillars(pybind11::array_t<float> points,
                              int maxPointsPerPillar, int maxPillars,
                              float xStep, float yStep, float xMin, float xMax,
                              float yMin, float yMax, float zMin, float zMax,
                              bool printTime, float minDistance) {
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  if (points.ndim() != 2 || points.shape()[1] != 4) {
    throw std::runtime_error(
        "numpy array with shape (n, 4) expected (n being the number of "
        "points)");
  }

  std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<PillarPoint>,
                     IntPairHash>
      map;

  for (int i = 0; i < points.shape()[0]; ++i) {
    if ((points.at(i, 0) < xMin) || (points.at(i, 0) >= xMax) ||
        (points.at(i, 1) < yMin) || (points.at(i, 1) >= yMax) ||
        (points.at(i, 2) < zMin) || (points.at(i, 2) >= zMax) ||
        minDistance > 0 && (std::pow(points.at(i, 0), 2) + std::pow(points.at(i, 1), 2)) < std::pow(minDistance, 2)) 
    {
      continue;
    }

    auto xIndex =
        static_cast<uint32_t>(std::floor((points.at(i, 0) - xMin) / xStep));
    auto yIndex =
        static_cast<uint32_t>(std::floor((points.at(i, 1) - yMin) / yStep));

    PillarPoint p = {
        points.at(i, 0),
        points.at(i, 1),
        points.at(i, 2),
        clamp(points.at(i, 3), 0.0f, 1.0f),
        0,
        0,
        0,
    };

    map[{xIndex, yIndex}].emplace_back(p);
  }

  pybind11::array_t<float> tensor;
  pybind11::array_t<int> indices;

  tensor.resize({1, maxPillars, maxPointsPerPillar, 9});
  // Have to be careful about unitialized pillars if num pillars < max_pillars.
  // All unitialized pillars will be written into (batch_id, 0, 0) as no empty
  // pillar is known.
  // TODO (mgier) find one random unitialized x, y pair and write all empty
  // pillars
  // into there.
  indices.resize({1, maxPillars, 3});
  // For now do zero padding on both ends.
  // Pillar tensor.
  pybind11::buffer_info tensor_buffer = tensor.request();
  float* ptr_tensor = (float*)tensor_buffer.ptr;
  for (size_t idx = 0; idx < 1 * maxPillars * maxPointsPerPillar * 9; idx++) {
    ptr_tensor[idx] = 0.0;
  }
  // Indices.
  pybind11::buffer_info indices_buffer = indices.request();
  int* ptr_indices = (int*)indices_buffer.ptr;
  for (size_t idx = 0; idx < 1 * maxPillars * 3; idx++) {
    ptr_indices[idx] = 0;
  }

  int pillarId = 0;
  for (auto& pair : map) {
    if (pillarId >= maxPillars) {
      break;
    }

    float xMean = 0;
    float yMean = 0;
    float zMean = 0;
    for (const auto& p : pair.second) {
      xMean += p.x;
      yMean += p.y;
      zMean += p.z;
    }
    xMean /= pair.second.size();
    yMean /= pair.second.size();
    zMean /= pair.second.size();

    for (auto& p : pair.second) {
      p.xc = p.x - xMean;
      p.yc = p.y - yMean;
      p.zc = p.z - zMean;
    }

    auto xIndex = static_cast<int>(std::floor((xMean - xMin) / xStep));
    auto yIndex = static_cast<int>(std::floor((yMean - yMin) / yStep));
    indices.mutable_at(0, pillarId, 1) = xIndex;
    indices.mutable_at(0, pillarId, 2) = yIndex;

    int pointId = 0;
    for (const auto& p : pair.second) {
      if (pointId >= maxPointsPerPillar) {
        break;
      }

      // Chapter 2.1 https://arxiv.org/pdf/1812.05784.pdf. 9 dimensional input
      // to network.
      tensor.mutable_at(0, pillarId, pointId, 0) = p.x;
      tensor.mutable_at(0, pillarId, pointId, 1) = p.y;
      tensor.mutable_at(0, pillarId, pointId, 2) = p.z;
      tensor.mutable_at(0, pillarId, pointId, 3) = p.intensity;
      // Subscript c refers to the distance to the arithmetic mean of all points
      // in the pillar.
      tensor.mutable_at(0, pillarId, pointId, 4) = p.xc;
      tensor.mutable_at(0, pillarId, pointId, 5) = p.yc;
      tensor.mutable_at(0, pillarId, pointId, 6) = p.zc;
      // Subscript p offset from pillar center in x and y.
      tensor.mutable_at(0, pillarId, pointId, 7) =
          p.x - (xIndex * xStep + xMin);
      tensor.mutable_at(0, pillarId, pointId, 8) =
          p.y - (yIndex * yStep + yMin);
      // TODO remove this as there is no pillar center -> no features learned
      // through this.
      // tensor.mutable_at(0, pillarId, pointId, 2) = p.z - zMid;

      pointId++;
    }

    pillarId++;
  }

  pybind11::tuple result = pybind11::make_tuple(tensor, indices);

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (printTime)
    std::cout << "createPillars took: " << static_cast<float>(duration) / 1e6
              << " seconds" << std::endl;

  return result;
}

struct BoundingBox3D {
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  // For anchors only!
  float base_yaw = 0.0;
  float classId;
};

struct BoundingBox3DVel {
  float x;
  float y;
  float z;
  float length;
  float width;
  float height;
  float yaw;
  float x_vel;
  float y_vel;
  // For anchors only!
  float base_yaw = 0.0;
  float classId;
};

struct Point2D {
  float x;
  float y;
};

typedef std::vector<Point2D> Polyline2D;

// Returns x-value of point of intersection of two lines
float xIntersect(float x1, float y1, float x2, float y2, float x3, float y3,
                 float x4, float y4) {
  float num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4);
  float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  return num / den;
}

// Returns y-value of point of intersection of two lines
float yIntersect(float x1, float y1, float x2, float y2, float x3, float y3,
                 float x4, float y4) {
  float num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4);
  float den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  return num / den;
}

// Returns area of polygon using the shoelace method
float polygonArea(const Polyline2D& polygon) {
  float area = 0.0;

  size_t j = polygon.size() - 1;
  for (size_t i = 0; i < polygon.size(); i++) {
    area += (polygon[j].x + polygon[i].x) * (polygon[j].y - polygon[i].y);
    j = i;  // j is previous vertex to i
  }

  return std::abs(area / 2.0);  // Return absolute value
}

float rotatedX(float x, float y, float angle) {
  return x * std::cos(angle) - y * std::sin(angle);
}

float rotatedY(float x, float y, float angle) {
  return x * std::sin(angle) + y * std::cos(angle);
}

// Construct bounding box in 2D, coordinates are returned in clockwise order
template<typename BoxType>
Polyline2D boundingBox3DToTopDown(const BoxType& box1) {
  Polyline2D box;
  box.push_back(
      {rotatedX(-0.5 * box1.length, 0.5 * box1.width, box1.yaw) + box1.x,
       rotatedY(-0.5 * box1.length, 0.5 * box1.width, box1.yaw) + box1.y});

  box.push_back(
      {rotatedX(0.5 * box1.length, 0.5 * box1.width, box1.yaw) + box1.x,
       rotatedY(0.5 * box1.length, 0.5 * box1.width, box1.yaw) + box1.y});

  box.push_back(
      {rotatedX(0.5 * box1.length, -0.5 * box1.width, box1.yaw) + box1.x,
       rotatedY(0.5 * box1.length, -0.5 * box1.width, box1.yaw) + box1.y});

  box.push_back(
      {rotatedX(-0.5 * box1.length, -0.5 * box1.width, box1.yaw) + box1.x,
       rotatedY(-0.5 * box1.length, -0.5 * box1.width, box1.yaw) + box1.y});

  return box;
}

// This functions clips all the edges w.r.t one Clip edge of clipping area
// Returns a clipped polygon...
Polyline2D clip(const Polyline2D& poly_points, float x1, float y1, float x2,
                float y2) {
  Polyline2D new_points;

  for (size_t i = 0; i < poly_points.size(); i++) {
    // (ix,iy),(kx,ky) are the co-ordinate values of the points
    // i and k form a line in polygon
    size_t k = (i + 1) % poly_points.size();
    float ix = poly_points[i].x, iy = poly_points[i].y;
    float kx = poly_points[k].x, ky = poly_points[k].y;

    // Calculating position of first point w.r.t. clipper line
    float i_pos = (x2 - x1) * (iy - y1) - (y2 - y1) * (ix - x1);

    // Calculating position of second point w.r.t. clipper line
    float k_pos = (x2 - x1) * (ky - y1) - (y2 - y1) * (kx - x1);

    // Case 1 : When both points are inside
    if (i_pos < 0 && k_pos < 0) {
      // Only second point is added
      new_points.push_back({kx, ky});
    }

    // Case 2: When only first point is outside
    else if (i_pos >= 0 && k_pos < 0) {
      // Point of intersection with edge
      // and the second point is added
      new_points.push_back({xIntersect(x1, y1, x2, y2, ix, iy, kx, ky),
                            yIntersect(x1, y1, x2, y2, ix, iy, kx, ky)});
      new_points.push_back({kx, ky});

    }

    // Case 3: When only second point is outside
    else if (i_pos < 0 && k_pos >= 0) {
      // Only point of intersection with edge is added
      new_points.push_back({xIntersect(x1, y1, x2, y2, ix, iy, kx, ky),
                            yIntersect(x1, y1, x2, y2, ix, iy, kx, ky)});

    }
    // Case 4: When both points are outside
    else {
      // No points are added
    }
  }

  return new_points;
}

// Implements Sutherland–Hodgman algorithm
// Returns a polygon with the intersection between two polygons.
Polyline2D sutherlandHodgmanClip(const Polyline2D& poly_points_vector,
                                 const Polyline2D& clipper_points) {
  Polyline2D clipped_poly_points_vector = poly_points_vector;
  for (size_t i = 0; i < clipper_points.size(); i++) {
    size_t k =
        (i + 1) % clipper_points.size();  // i and k are two consecutive indexes

    // We pass the current array of vertices, and the end points of the selected
    // clipper line
    clipped_poly_points_vector =
        clip(clipped_poly_points_vector, clipper_points[i].x,
             clipper_points[i].y, clipper_points[k].x, clipper_points[k].y);
  }
  return clipped_poly_points_vector;
}

// Calculates the IOU between two bounding boxes.
template <typename BoxType1, typename BoxType2>
float iou(const BoxType1& box1, const BoxType2& box2) {
  const auto& box_as_vector = boundingBox3DToTopDown(box1);
  const auto& box_as_vector_2 = boundingBox3DToTopDown(box2);
  const auto& clipped_vector =
      sutherlandHodgmanClip(box_as_vector, box_as_vector_2);

  float area_poly1 = polygonArea(box_as_vector);
  float area_poly2 = polygonArea(box_as_vector_2);
  float area_overlap = polygonArea(clipped_vector);

  return area_overlap / (area_poly1 + area_poly2 - area_overlap);
}

int clip(int n, int lower, int upper) {
  return std::max(lower, std::min(n, upper));
}

pybind11::array_t<float> createPillarsTarget(
    const pybind11::array_t<float>& objectPositions,
    const pybind11::array_t<float>& objectDimensions,
    const pybind11::array_t<float>& objectYaws,
    const pybind11::array_t<int>& objectClassIds,
    const pybind11::array_t<float>& anchorDimensions,
    const pybind11::array_t<float>& anchorZHeights,
    const pybind11::array_t<float>& anchorYaws, float positiveThreshold,
    float negativeThreshold, float angle_threshold, int nbClasses,
    int downscalingFactor, float xStep, float yStep, float xMin, float xMax,
    float yMin, float yMax, float zMin, float zMax, bool printTime = false) {
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  const auto xSize =
      static_cast<int>(std::floor((xMax - xMin) / (xStep * downscalingFactor)));
  const auto ySize =
      static_cast<int>(std::floor((yMax - yMin) / (yStep * downscalingFactor)));

  const int nbAnchors = anchorDimensions.shape()[0];

  if (nbAnchors <= 0) {
    throw std::runtime_error("Anchor length is zero");
  }

  const int nbObjects = objectDimensions.shape()[0];
  if (nbObjects <= 0) {
    throw std::runtime_error("Object length is zero");
  }

  // parse numpy arrays
  std::vector<BoundingBox3D> anchorBoxes = {};
  std::vector<float> anchorDiagonals;
  for (int i = 0; i < nbAnchors; ++i) {
    BoundingBox3D anchorBox = {};
    anchorBox.x = 0;
    anchorBox.y = 0;
    anchorBox.length = anchorDimensions.at(i, 0);
    anchorBox.width = anchorDimensions.at(i, 1);
    anchorBox.height = anchorDimensions.at(i, 2);
    anchorBox.z = anchorZHeights.at(i);
    anchorBox.yaw = anchorYaws.at(i);
    anchorBox.base_yaw = anchorBox.yaw;
    anchorBoxes.emplace_back(anchorBox);

    anchorDiagonals.emplace_back(std::sqrt(std::pow(anchorBox.width, 2) +
                                           std::pow(anchorBox.length, 2)));
  }

  std::vector<BoundingBox3D> labelBoxes = {};
  for (int i = 0; i < nbObjects; ++i) {
    float x = objectPositions.at(i, 0);
    float y = objectPositions.at(i, 1);
    // Exclude equality on max values since this does not find into the
    // discretized grid.
    if (x < xMin | x >= xMax | y < yMin | y >= yMax) {
      continue;
    }
    BoundingBox3D labelBox = {};
    labelBox.x = x;
    labelBox.y = y;
    labelBox.z = objectPositions.at(i, 2);
    labelBox.length = objectDimensions.at(i, 0);
    labelBox.width = objectDimensions.at(i, 1);
    labelBox.height = objectDimensions.at(i, 2);
    labelBox.yaw = objectYaws.at(i);
    labelBox.classId = objectClassIds.at(i);
    labelBoxes.emplace_back(labelBox);
  }

  pybind11::array_t<float> tensor;
  tensor.resize({nbObjects, xSize, ySize, nbAnchors, 10});

  pybind11::buffer_info tensor_buffer = tensor.request();
  float* ptr1 = (float*)tensor_buffer.ptr;
  for (size_t idx = 0; idx < nbObjects * xSize * ySize * nbAnchors * 10;
       idx++) {
    ptr1[idx] = 0;
  }

  int objectCount = 0;
  if (printTime) {
    std::cout << "Received " << labelBoxes.size() << " objects" << std::endl;
  }
  for (const auto& labelBox : labelBoxes) {
    // zone-in on potential spatial area of interest
    float objectDiameter =
        std::sqrt(std::pow(labelBox.width, 2) + std::pow(labelBox.length, 2));
    const auto xOffset = static_cast<int>(
        std::ceil(objectDiameter / (xStep * downscalingFactor)));
    const auto yOffset = static_cast<int>(
        std::ceil(objectDiameter / (yStep * downscalingFactor)));
    const auto xC = static_cast<int>(
        std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
    const auto yC = static_cast<int>(
        std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));

    const auto xStart = clip(xC - xOffset, 0, xSize);
    const auto yStart = clip(yC - yOffset, 0, ySize);
    const auto xEnd = clip(xC + xOffset, 0, xSize);
    const auto yEnd = clip(yC + yOffset, 0, ySize);

    float maxIou = 0;
    BoundingBox3D bestAnchor = {};
    int bestAnchorId = 0;
    for (int xId = xStart; xId < xEnd; xId++) {
      const float x = xId * xStep * downscalingFactor + xMin;

      for (int yId = yStart; yId < yEnd; yId++) {
        const float y = yId * yStep * downscalingFactor + yMin;
        int anchorCount = 0;
        for (auto& anchorBox : anchorBoxes) {
          anchorBox.x = x;
          anchorBox.y = y;

          // If the angle is within the allowed threshold, rotate it.
          const float delta_yaw_no =
              std::fmod(labelBox.yaw - anchorBox.base_yaw, M_PI);
          if (std::abs(delta_yaw_no) < angle_threshold ||
              (M_PI - std::abs(delta_yaw_no)) < angle_threshold) {
            // Override the anchor yaw to "label yaw" in order to sufficiently
            // cover roated boxes between anchors.
            anchorBox.yaw = labelBox.yaw;
          } else {
            // Otherwise make sure the anchor rotation is set to baseline yaw.
            anchorBox.yaw = anchorBox.base_yaw;
          }

          const float iouOverlap = iou(anchorBox, labelBox);

          if (maxIou < iouOverlap) {
            maxIou = iouOverlap;
            bestAnchor = anchorBox;
            bestAnchorId = anchorCount;
          }

          if (iouOverlap > positiveThreshold) {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 1;

            auto diag = anchorDiagonals[anchorCount];
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 1) =
                (labelBox.x - anchorBox.x) / diag;
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 2) =
                (labelBox.y - anchorBox.y) / diag;
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 3) =
                (labelBox.z - anchorBox.z) / anchorBox.height;

            tensor.mutable_at(objectCount, xId, yId, anchorCount, 4) =
                std::log(labelBox.length / anchorBox.length);
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 5) =
                std::log(labelBox.width / anchorBox.width);
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 6) =
                std::log(labelBox.height / anchorBox.height);

            // Reduce angle to an interval of [-pi/2, pi/2] so that
            // the sine is invertible. The angle is the delta angle
            // of a *not oriented* box.
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 7) = std::sin(
                std::abs(delta_yaw_no) > M_PI_2 ? -delta_yaw_no : delta_yaw_no);
            // Encode whether the heading of the vehicle has to be
            // flipped around. The heading must be flipped if
            // delta angle > 90 and < 270. The angle is an oriented
            // angle.
            // TODO (gier) can this be easier? just take the delta_yaw_no?
            const float delta_yaw_o =
                std::fmod(labelBox.yaw - anchorBox.base_yaw, 2 * M_PI);
            if (std::abs(delta_yaw_o) < M_PI_2 &&
                std::abs(delta_yaw_o) > 1.5 * M_PI) {
              tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 1;
            } else {
              tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 0;
            }

            tensor.mutable_at(objectCount, xId, yId, anchorCount, 9) =
                labelBox.classId;

          } else if (iouOverlap < negativeThreshold) {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 0;
          } else {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = -1;
          }

          anchorCount++;
        }
      }
    }

    if (maxIou < positiveThreshold) {
      if (printTime) {
        std::cout << "\nThere was no sufficiently overlapping anchor anywhere "
                     "for object "
                  << objectCount << std::endl;
        std::cout << "Best IOU was " << maxIou
                  << ". Adding the best location regardless of threshold."
                  << std::endl;
      }

      const auto xId_0 = static_cast<int>(
          std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
      const auto yId_0 = static_cast<int>(
          std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));

      for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
          // Get current x and y id from xId_0 and yId_0.
          const auto xId = clip(xId_0 + dx, 0, xSize - 1);
          const auto yId = clip(yId_0 + dy, 0, ySize - 1);

          if (dx == 0 && dy == 0) {
            // Set as occupied for the actual best anchor.

            const float diag = std::sqrt(std::pow(bestAnchor.width, 2) +
                                         std::pow(bestAnchor.length, 2));
            // The best anchor can be at various locations in case the object is
            // large and covering multiple boxes completely (e.g. bus).
            // Assume that the best anchor is still the one with the right shape
            // at this location.
            const float x = xId * xStep * downscalingFactor + xMin;
            const float y = yId * yStep * downscalingFactor + yMin;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 0) = 1;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 1) =
                (labelBox.x - x) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 2) =
                (labelBox.y - y) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 3) =
                (labelBox.z - bestAnchor.z) / bestAnchor.height;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 4) =
                std::log(labelBox.length / bestAnchor.length);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 5) =
                std::log(labelBox.width / bestAnchor.width);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 6) =
                std::log(labelBox.height / bestAnchor.height);

            // Reduce angle to an interval of [-pi/2, pi/2] so that
            // the sine is invertible. The angle is the delta angle
            // of a not oriented box.
            const float delta_yaw_no =
                std::fmod(labelBox.yaw - bestAnchor.base_yaw, M_PI);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 7) =
                std::sin(std::abs(delta_yaw_no) > M_PI_2 ? -delta_yaw_no
                                                         : delta_yaw_no);
            // Encode whether the heading of the vehicle has to be
            // flipped around. The heading must be flipped if
            // delta angle > 90 and < 270. The angle is an oriented angle.
            const float delta_yaw_o =
                std::fmod(labelBox.yaw - bestAnchor.base_yaw, 2 * M_PI);
            if (std::abs(delta_yaw_o) < M_PI_2 &&
                std::abs(delta_yaw_o) > 1.5 * M_PI) {
              tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 1;
            } else {
              tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 0;
            }
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 9) =
                labelBox.classId;

          } else if (xId_0 + dx >= 0 && xId_0 + dx < xSize && yId_0 + dy >= 0 &&
                     yId_0 + dy < ySize) {
            // Make sure the the neighboring field would still be in the range.
            // Otherwise the clipped value could override
            // the positive anchor.

            // Set to -1 in order to do not penalize for scores in the
            // surrounding of the object.
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 0) = -1;
          }
        }
      }
    } else {
      if (printTime) {
        std::cout << "\nAt least 1 anchor was positively matched for object "
                  << objectCount << std::endl;
        std::cout << "Best IOU was " << maxIou << "." << std::endl;
      }
    }

    objectCount++;
  }

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (printTime)
    std::cout << "createPillarsTarget took: "
              << static_cast<float>(duration) / 1e6 << " seconds" << std::endl;

  return tensor;
}

pybind11::array_t<float> createPillarsTargetWithVelocity(
    const pybind11::array_t<float>& objectPositions,
    const pybind11::array_t<float>& objectDimensions,
    const pybind11::array_t<float>& objectYaws,
    const pybind11::array_t<float>& objectVelocities,
    const pybind11::array_t<int>& objectClassIds,
    const pybind11::array_t<float>& anchorDimensions,
    const pybind11::array_t<float>& anchorZHeights,
    const pybind11::array_t<float>& anchorYaws, float positiveThreshold,
    float negativeThreshold, float angle_threshold, int nbClasses,
    int downscalingFactor, float xStep, float yStep, float xMin, float xMax,
    float yMin, float yMax, float zMin, float zMax, bool printTime = false) {
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  const auto xSize =
      static_cast<int>(std::floor((xMax - xMin) / (xStep * downscalingFactor)));
  const auto ySize =
      static_cast<int>(std::floor((yMax - yMin) / (yStep * downscalingFactor)));

  const int nbAnchors = anchorDimensions.shape()[0];

  if (nbAnchors <= 0) {
    throw std::runtime_error("Anchor length is zero");
  }

  const int nbObjects = objectDimensions.shape()[0];
  if (nbObjects <= 0) {
    throw std::runtime_error("Object length is zero");
  }

  // parse numpy arrays
  std::vector<BoundingBox3DVel> anchorBoxes = {};
  std::vector<float> anchorDiagonals;
  for (int i = 0; i < nbAnchors; ++i) {
    BoundingBox3DVel anchorBox = {};
    anchorBox.x = 0;
    anchorBox.y = 0;
    anchorBox.length = anchorDimensions.at(i, 0);
    anchorBox.width = anchorDimensions.at(i, 1);
    anchorBox.height = anchorDimensions.at(i, 2);
    anchorBox.z = anchorZHeights.at(i);
    anchorBox.yaw = anchorYaws.at(i);
    anchorBox.x_vel = objectVelocities.at(i, 0);
    anchorBox.y_vel = objectVelocities.at(i, 1);
    anchorBox.base_yaw = anchorBox.yaw;
    anchorBoxes.emplace_back(anchorBox);

    anchorDiagonals.emplace_back(std::sqrt(std::pow(anchorBox.width, 2) +
                                           std::pow(anchorBox.length, 2)));
  }

  std::vector<BoundingBox3DVel> labelBoxes = {};
  for (int i = 0; i < nbObjects; ++i) {
    float x = objectPositions.at(i, 0);
    float y = objectPositions.at(i, 1);
    // Exclude equality on max values since this does not find into the
    // discretized grid.
    if (x < xMin | x >= xMax | y < yMin | y >= yMax) {
      continue;
    }
    BoundingBox3DVel labelBox = {};
    labelBox.x = x;
    labelBox.y = y;
    labelBox.z = objectPositions.at(i, 2);
    labelBox.length = objectDimensions.at(i, 0);
    labelBox.width = objectDimensions.at(i, 1);
    labelBox.height = objectDimensions.at(i, 2);
    labelBox.yaw = objectYaws.at(i);
    labelBox.x_vel = objectVelocities.at(i, 0);
    labelBox.y_vel = objectVelocities.at(i, 1);
    labelBox.classId = objectClassIds.at(i);
    labelBoxes.emplace_back(labelBox);
  }

  pybind11::array_t<float> tensor;
  tensor.resize({nbObjects, xSize, ySize, nbAnchors, 12});

  pybind11::buffer_info tensor_buffer = tensor.request();
  float* ptr1 = (float*)tensor_buffer.ptr;
  for (size_t idx = 0; idx < nbObjects * xSize * ySize * nbAnchors * 10;
       idx++) {
    ptr1[idx] = 0;
  }

  int objectCount = 0;
  if (printTime) {
    std::cout << "Received " << labelBoxes.size() << " objects" << std::endl;
  }
  for (const auto& labelBox : labelBoxes) {
    // zone-in on potential spatial area of interest
    float objectDiameter =
        std::sqrt(std::pow(labelBox.width, 2) + std::pow(labelBox.length, 2));
    const auto xOffset = static_cast<int>(
        std::ceil(objectDiameter / (xStep * downscalingFactor)));
    const auto yOffset = static_cast<int>(
        std::ceil(objectDiameter / (yStep * downscalingFactor)));
    const auto xC = static_cast<int>(
        std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
    const auto yC = static_cast<int>(
        std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));

    const auto xStart = clip(xC - xOffset, 0, xSize);
    const auto yStart = clip(yC - yOffset, 0, ySize);
    const auto xEnd = clip(xC + xOffset, 0, xSize);
    const auto yEnd = clip(yC + yOffset, 0, ySize);

    float maxIou = 0;
    BoundingBox3DVel bestAnchor = {};
    int bestAnchorId = 0;
    for (int xId = xStart; xId < xEnd; xId++) {
      const float x = xId * xStep * downscalingFactor + xMin;

      for (int yId = yStart; yId < yEnd; yId++) {
        const float y = yId * yStep * downscalingFactor + yMin;
        int anchorCount = 0;
        for (auto& anchorBox : anchorBoxes) {
          anchorBox.x = x;
          anchorBox.y = y;

          // If the angle is within the allowed threshold, rotate it.
          const float delta_yaw_no =
              std::fmod(labelBox.yaw - anchorBox.base_yaw, M_PI);
          if (std::abs(delta_yaw_no) < angle_threshold ||
              (M_PI - std::abs(delta_yaw_no)) < angle_threshold) {
            // Override the anchor yaw to "label yaw" in order to sufficiently
            // cover roated boxes between anchors.
            anchorBox.yaw = labelBox.yaw;
          } else {
            // Otherwise make sure the anchor rotation is set to baseline yaw.
            anchorBox.yaw = anchorBox.base_yaw;
          }

          const float iouOverlap = iou(anchorBox, labelBox);

          if (maxIou < iouOverlap) {
            maxIou = iouOverlap;
            bestAnchor = anchorBox;
            bestAnchorId = anchorCount;
          }

          if (iouOverlap > positiveThreshold) {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 1;

            auto diag = anchorDiagonals[anchorCount];
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 1) =
                (labelBox.x - anchorBox.x) / diag;
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 2) =
                (labelBox.y - anchorBox.y) / diag;
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 3) =
                (labelBox.z - anchorBox.z) / anchorBox.height;

            tensor.mutable_at(objectCount, xId, yId, anchorCount, 4) =
                std::log(labelBox.length / anchorBox.length);
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 5) =
                std::log(labelBox.width / anchorBox.width);
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 6) =
                std::log(labelBox.height / anchorBox.height);

            // Reduce angle to an interval of [-pi/2, pi/2] so that
            // the sine is invertible. The angle is the delta angle
            // of a *not oriented* box.
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 7) = std::sin(
                std::abs(delta_yaw_no) > M_PI_2 ? -delta_yaw_no : delta_yaw_no);
            // Encode whether the heading of the vehicle has to be
            // flipped around. The heading must be flipped if
            // delta angle > 90 and < 270. The angle is an oriented
            // angle.
            // TODO (gier) can this be easier? just take the delta_yaw_no?
            const float delta_yaw_o =
                std::fmod(labelBox.yaw - anchorBox.base_yaw, 2 * M_PI);
            if (std::abs(delta_yaw_o) < M_PI_2 &&
                std::abs(delta_yaw_o) > 1.5 * M_PI) {
              tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 1;
            } else {
              tensor.mutable_at(objectCount, xId, yId, anchorCount, 8) = 0;
            }

            tensor.mutable_at(objectCount, xId, yId, anchorCount, 9) =
                labelBox.classId;

            tensor.mutable_at(objectCount, xId, yId, anchorCount, 10) = labelBox.x_vel;
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 11) = labelBox.y_vel;

          } else if (iouOverlap < negativeThreshold) {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = 0;
          } else {
            tensor.mutable_at(objectCount, xId, yId, anchorCount, 0) = -1;
          }

          anchorCount++;
        }
      }
    }

    if (maxIou < positiveThreshold) {
      if (printTime) {
        std::cout << "\nThere was no sufficiently overlapping anchor anywhere "
                     "for object "
                  << objectCount << std::endl;
        std::cout << "Best IOU was " << maxIou
                  << ". Adding the best location regardless of threshold."
                  << std::endl;
      }

      const auto xId_0 = static_cast<int>(
          std::floor((labelBox.x - xMin) / (xStep * downscalingFactor)));
      const auto yId_0 = static_cast<int>(
          std::floor((labelBox.y - yMin) / (yStep * downscalingFactor)));

      for (int dx = -2; dx <= 2; ++dx) {
        for (int dy = -2; dy <= 2; ++dy) {
          // Get current x and y id from xId_0 and yId_0.
          const auto xId = clip(xId_0 + dx, 0, xSize - 1);
          const auto yId = clip(yId_0 + dy, 0, ySize - 1);

          if (dx == 0 && dy == 0) {
            // Set as occupied for the actual best anchor.

            const float diag = std::sqrt(std::pow(bestAnchor.width, 2) +
                                         std::pow(bestAnchor.length, 2));
            // The best anchor can be at various locations in case the object is
            // large and covering multiple boxes completely (e.g. bus).
            // Assume that the best anchor is still the one with the right shape
            // at this location.
            const float x = xId * xStep * downscalingFactor + xMin;
            const float y = yId * yStep * downscalingFactor + yMin;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 0) = 1;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 1) =
                (labelBox.x - x) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 2) =
                (labelBox.y - y) / diag;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 3) =
                (labelBox.z - bestAnchor.z) / bestAnchor.height;

            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 4) =
                std::log(labelBox.length / bestAnchor.length);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 5) =
                std::log(labelBox.width / bestAnchor.width);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 6) =
                std::log(labelBox.height / bestAnchor.height);

            // Reduce angle to an interval of [-pi/2, pi/2] so that
            // the sine is invertible. The angle is the delta angle
            // of a not oriented box.
            const float delta_yaw_no =
                std::fmod(labelBox.yaw - bestAnchor.base_yaw, M_PI);
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 7) =
                std::sin(std::abs(delta_yaw_no) > M_PI_2 ? -delta_yaw_no
                                                         : delta_yaw_no);
            // Encode whether the heading of the vehicle has to be
            // flipped around. The heading must be flipped if
            // delta angle > 90 and < 270. The angle is an oriented angle.
            const float delta_yaw_o =
                std::fmod(labelBox.yaw - bestAnchor.base_yaw, 2 * M_PI);
            if (std::abs(delta_yaw_o) < M_PI_2 &&
                std::abs(delta_yaw_o) > 1.5 * M_PI) {
              tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 1;
            } else {
              tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 8) = 0;
            }
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 9) =
                labelBox.classId;
                
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 10) = labelBox.x_vel;
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 11) = labelBox.y_vel;

          } else if (xId_0 + dx >= 0 && xId_0 + dx < xSize && yId_0 + dy >= 0 &&
                     yId_0 + dy < ySize) {
            // Make sure the the neighboring field would still be in the range.
            // Otherwise the clipped value could override
            // the positive anchor.

            // Set to -1 in order to do not penalize for scores in the
            // surrounding of the object.
            tensor.mutable_at(objectCount, xId, yId, bestAnchorId, 0) = -1;
          }
        }
      }
    } else {
      if (printTime) {
        std::cout << "\nAt least 1 anchor was positively matched for object "
                  << objectCount << std::endl;
        std::cout << "Best IOU was " << maxIou << "." << std::endl;
      }
    }

    objectCount++;
  }

  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (printTime)
    std::cout << "createPillarsTarget took: "
              << static_cast<float>(duration) / 1e6 << " seconds" << std::endl;

  return tensor;
}


PYBIND11_MODULE(point_pillars, m) {
  m.def("createPillars", &createPillars,
        "Runs function to create point pillars input tensors",
        pybind11::arg("points"), pybind11::arg("maxPointsPerPillar"), pybind11::arg("maxPillars"),
        pybind11::arg("xStep"), pybind11::arg("yStep"), pybind11::arg("xMin"), pybind11::arg("xMax"),
        pybind11::arg("yMin"), pybind11::arg("yMax"), pybind11::arg("zMin"), pybind11::arg("zMax"),
        pybind11::arg("printTime") = false, pybind11::arg("minDistance") = -1.0);
  m.def("createPillarsTarget", &createPillarsTarget,
        "Runs function to create point pillars output ground truth");
  m.def("createPillarsTargetWithVelocity", &createPillarsTargetWithVelocity,
        "Runs function to create point pillars output ground truth with velocity");
}
