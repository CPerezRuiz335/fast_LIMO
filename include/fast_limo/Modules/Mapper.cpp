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



IKDTree::IKDTree() {
  Config& config = Config::getInstance();
  map = KD_TREE<MapPoint>::Ptr(
          new KD_TREE<MapPoint>(config.ikfom.mapping.ikdtree.delete_param,
                                config.ikfom.mapping.ikdtree.balance_param,
                                config.ikfom.mapping.ikdtree.voxel_size));
}
    
bool IKDTree::exists() {
  return map->size() > 0;
}

int IKDTree::size() {
  return map->size();
}

void IKDTree::add(PointCloudT::Ptr& pc, bool downsample) {
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
}


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

