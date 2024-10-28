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

// #include "octree2/Octree.h"
#include "ikd-Tree/ikd_Tree/ikd_Tree.h"

#include "fast_limo/Objects/State.hpp"
#include "fast_limo/Utils/Config.hpp"
#include "fast_limo/Utils/PCL.hpp"

using namespace fast_limo;

namespace fast_limo {
  class IKDTree {

    public:
      KD_TREE<MapPoint>::Ptr map;

      IKDTree();

      bool exists();
      int size();

      void knn(const MapPoint& p,
               int& k,
               MapPoints& near_points,
               std::vector<float>& sqDist);

      void add(PointCloudT::Ptr&, bool downsample=true);

    private:
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

typedef IKDTree Mapper;

}

#endif