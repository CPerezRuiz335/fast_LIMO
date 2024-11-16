/*
 Copyright (c) 2024 Oriol Martínez @fetty31

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "fast_limo/Modules/Mapper.hpp"



IKDTree::IKDTree() : last_map_time_(-1.),
                   num_threads_(1) {

  map = KD_TREE<MapPoint>::Ptr(new KD_TREE<MapPoint>(0.3, 0.6, 0.2));

  // Init cfg values
  config.NUM_MATCH_POINTS = 5;
  config.MAX_NUM_PC2MATCH = 1.e+4;
  config.MAX_DIST_PLANE   = 2.0;
  config.PLANE_THRESHOLD  = 5.e-2;
  config.local_mapping    = true;

  config.ikdtree.cube_size = 300.0;
  config.ikdtree.rm_range  = 200.0;
}
    
bool IKDTree::exists() {
  return map->size() > 0;
}

int IKDTree::size() {
  return map->size();
}

double IKDTree::last_time() {
  return last_map_time_;
}


void IKDTree::add(PointCloudT::Ptr& pc, double time, bool downsample) {
  if (pc->points.size() < 1)
    return;

  // If map doesn't exists, build one
  if (not exists()) {
    build(pc);
  } else {
    MapPoints map_vec;
    map_vec.reserve(pc->points.size());

    for (int i = 0; i < pc->points.size(); i++)
      map_vec.emplace_back(pc->points[i].x, pc->points[i].y, pc->points[i].z);

    map->Add_Points(map_vec, downsample);
  }

  last_map_time_ = time;
}

// private

void IKDTree::build(PointCloudT::Ptr& pc) {
  MapPoints map_vec;
  map_vec.reserve(pc->points.size());

  for(int i = 0; i < pc->points.size (); i++)
    map_vec.emplace_back(pc->points[i].x, pc->points[i].y, pc->points[i].z);

  this->map->Build(map_vec);
}

void IKDTree::knn(const MapPoint& p,
         int& k,
         MapPoints& near_points,
         std::vector<float>& sqDist) {

  map->Nearest_Search(p, k, near_points, sqDist);
}









BonxaiTree::BonxaiTree() : map(0.2) {
  struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int>& t) const {
        auto [x, y, z] = t;
        return std::hash<int>()(x) ^ (std::hash<int>()(y) << 1) ^ (std::hash<int>()(z) << 2);
    }
  };

  // Define the six possible movement directions (left, right, up, down, forward, backward)
    const std::vector<std::tuple<int, int, int>> directions = {
      {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
      {-1, 0, -1},  {-1, 0, 0},  {-1, 0, 1},
      {-1, 1, -1},  {-1, 1, 0},  {-1, 1, 1},
      {0, -1, -1},  {0, -1, 0},  {0, -1, 1},
      {0, 0, -1},                {0, 0, 1},
      {0, 1, -1},   {0, 1, 0},   {0, 1, 1},
      {1, -1, -1},  {1, -1, 0},  {1, -1, 1},
      {1, 0, -1},   {1, 0, 0},   {1, 0, 1},
      {1, 1, -1},   {1, 1, 0},   {1, 1, 1}
    };

    // Queue for BFS traversal
    std::tuple<int, int, int> start = {0, 0, 0};
    std::queue<std::tuple<int, int, int>> q;
    q.push(start);

    // Unordered set to keep track of visited coordinates
    std::unordered_set<std::tuple<int, int, int>, TupleHash> visited;
    visited.insert(start);

    while (!q.empty()) {
      // Get the current position
      auto [x, y, z] = q.front();
      q.pop();

      if (x != 0 || y != 0 || z != 0) {
        Bonxai::CoordT coord;
        coord.x = x;
        coord.y = y;
        coord.z = z;
        coords.push_back(coord);
      }

      // Explore each direction
      for (const auto& [dx, dy, dz] : directions) {
        int new_x = x + dx;
        int new_y = y + dy;
        int new_z = z + dz;
        std::tuple<int, int, int> neighbor = {new_x, new_y, new_z};

        // If the neighbor hasn't been visited yet, add it to the queue and mark as visited
        if (visited.find(neighbor) == visited.end() 
            && std::max(std::max(abs(new_x), abs(new_y)), abs(new_z)) < 9) {
            q.push(neighbor);
            visited.insert(neighbor);
        }
      }
    }

    for (int i = 0; i <= 8*2+1; i++) {
      arr.push_back(i*i*i);
    }
}



bool BonxaiTree::exists() { return size() > 0; }
int BonxaiTree::size() {return (int)map.activeCellsCount(); }
double BonxaiTree::last_time() {return last_map_time_; };

void BonxaiTree::knn(const MapPoint& p,
          int& k,
          MapPoints& near_points,
          std::vector<float>& sqDist) {
  
  PriorityQueue q;

  Accessor accessor = map.createAccessor();
  Bonxai::CoordT pCoord = map.posToCoord(p.x, p.y, p.z);
  
  int j(0);
  int arr[5] = {3*3*3, 5*5*5, 7*7*7, 9*9*9, 11*11*11};

  for (int i = 0; i < coords.size(); i++) {
    Bonxai::CoordT nCoord = pCoord;
    nCoord += coords[i];
    MapPoint* n = accessor.value(nCoord);
    
    if (i == arr[j]) {
      if (q.size() >= 5)
        break;
      j++;
    }
    
    if (n == nullptr)
      continue;
      

    double sqDist = squaredDistance(n->x, n->y, n->z, p.x, p.y, p.z);
    q.emplace(sqDist, *n);

  }

  std::cout << "Q size " << q.size() << std::endl; 
  if (q.size() >= 5) {
    for (int i = 0; i < k; i++) {
      Task tmp = q.top();
      near_points.emplace_back(tmp.point);
      sqDist.emplace_back(tmp.distance);
      q.pop();
    }
  } 
    
}

inline float l2_dist_pow2(const float& x1, const float& y1, const float& z1,
                          const float& x2, const float& y2, const float& z2) {
  float dx = x2 - x1;
  float dy = y2 - y1;
  float dz = z2 - z1;
  return dx * dx + dy * dy + dz * dz;
}


void BonxaiTree::add(PointCloudT::Ptr& pc, double time, bool downsample) {
  static Eigen::Vector3f leaf_size(0.2, 0.2, 0.2);

  if (pc->points.size() < 1)
    return;

  MapPoints map_vec;
  map_vec.reserve(pc->points.size());

  for (int i = 0; i < pc->points.size(); i++)
    map_vec.emplace_back(pc->points[i].x, pc->points[i].y, pc->points[i].z);

  Accessor accessor = map.createAccessor();
  for (const auto& p : map_vec) {
      Bonxai::CoordT coord = map.posToCoord(p.x, p.y, p.z);
      MapPoint* pn = accessor.value( coord );
      
      if (true) {
        accessor.setValue( coord, p );
      } else {
        Eigen::Vector3f point = Eigen::Vector3f(p.x, p.y, p.z);
        Eigen::Vector3f centroid = Eigen::Vector3f(coord.x, coord.y, coord.z);
        Eigen::Vector3f sign_vec = point.unaryExpr([](float v) { 
            return (v >= 0.0 ? 1.f : -1.f); 
        });

        centroid += (leaf_size/2).cwiseProduct(sign_vec);

        if (l2_dist_pow2(centroid.x(), centroid.y(), centroid.z(), p.x, p.y, p.z) 
            < l2_dist_pow2(centroid.x(), centroid.y(), centroid.z(), pn->x, pn->y, pn->z))
          accessor.setValue( coord, p );
      }
    }

  last_map_time_ = time;
}

void BonxaiTree::build(PointCloudT::Ptr&) {
  return;
}