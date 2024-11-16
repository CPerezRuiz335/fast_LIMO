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

#ifndef __FASTLIMO_MAPPER_HPP__
#define __FASTLIMO_MAPPER_HPP__

#include <ctime>
#include <iomanip>
#include <future>
#include <ios>
#include <sys/times.h>
#include <sys/vtimes.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <string>

#include <climits>
#include <cmath>

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <tuple>
#include <unordered_set>

#include "octree2/Octree.h"
#include "ikd-Tree/ikd_Tree/ikd_Tree.h"

#include "fast_limo/Objects/State.hpp"
#include "fast_limo/Utils/Config.hpp"

#include "bonxai.hpp"

using namespace fast_limo;

namespace fast_limo {
  class IKDTree {

    public:
      KD_TREE<MapPoint>::Ptr map;

      Config::iKFoM::Mapping config;

      double last_map_time_;

      int num_threads_;;

      IKDTree();

      bool exists();
      int size();
      double last_time();

      void knn(const MapPoint& p,
               int& k,
               MapPoints& near_points,
               std::vector<float>& sqDist);

      void add(PointCloudT::Ptr&, double time, bool downsample=true);

      MapPoints flatten() {
        MapPoints out;

        map->flatten(map->Root_Node, out, NOT_RECORD);

        return out;
      }

    public:
      void build(PointCloudT::Ptr&);


    public:
      static IKDTree& getInstance() {
        static IKDTree* ikd = new IKDTree();
        return *ikd;
      }

    private:
      // Disable copy/move capabilities
      IKDTree(const IKDTree&) = delete;
      IKDTree(IKDTree&&) = delete;

      IKDTree& operator=(const IKDTree&) = delete;
      IKDTree& operator=(IKDTree&&) = delete;
  };

  inline MapPoint chooseClosest(const Bonxai::CoordT& centroid, const MapPoint& old, const MapPoint& new_) {
    auto squaredDistance = [](const Bonxai::CoordT& c, const MapPoint& v) {
        double dx = static_cast<double>(c.x) - v.x;
        double dy = static_cast<double>(c.y) - v.y;
        double dz = static_cast<double>(c.z) - v.z;
        return dx * dx + dy * dy + dz * dz;
    };

    double distOld = squaredDistance(centroid, old);
    double distNew = squaredDistance(centroid, new_);

    return (distOld <= distNew) ? old : new_;
}


  struct Task {
    double distance;
    MapPoint point;

    Task(double d, MapPoint& p) : distance(d), point(p) {}
  };

  struct CompareTask {
    bool operator()(const Task &a, const Task &b) const {
      return a.distance > b.distance;
    }
  };

  typedef std::priority_queue<Task, std::vector<Task>, CompareTask> PriorityQueue;

  inline float squaredDistance(const float& x1, const float& y1, const float& z1,
                          const float& x2, const float& y2, const float& z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return dx * dx + dy * dy + dz * dz;
  }
  typedef Bonxai::VoxelGrid<MapPoint>::Accessor Accessor;

  class BonxaiTree {
    public:
      std::vector<Bonxai::CoordT> coords;
      std::vector<int> arr;

      Bonxai::VoxelGrid<MapPoint> map;

      Config::iKFoM::Mapping config;

      double last_map_time_;

      int num_threads_;

      BonxaiTree();

      bool exists();
      int size();
      double last_time();

      void knn(const MapPoint& p,
               int& k,
               MapPoints& near_points,
               std::vector<float>& sqDist);

      void add(PointCloudT::Ptr& pc, double time, bool downsample=true);
      void add(MapPoints& pc) {
        Accessor accessor = map.createAccessor();
        for (const auto& p : pc) {
            Bonxai::CoordT coord = map.posToCoord(p.x, p.y, p.z);
            accessor.setValue( coord, p );
          }
      }

      void clear() {
        map.clear(Bonxai::CLEAR_MEMORY);
      }

    public:
      void build(PointCloudT::Ptr&);

    public:
      static BonxaiTree& getInstance() {
        static BonxaiTree* ikd = new BonxaiTree();
        return *ikd;
      }

    private:
      BonxaiTree(const BonxaiTree&) = delete;
      BonxaiTree(BonxaiTree&&) = delete;

      BonxaiTree& operator=(const BonxaiTree&) = delete;
      BonxaiTree& operator=(BonxaiTree&&) = delete;
  };


typedef IKDTree Mapper;


}

#endif