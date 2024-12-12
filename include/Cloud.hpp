#pragma once  

#include <vector>
#include <algorithm>
#include <execution>

#include <boost/circular_buffer.hpp>

#include "PCL.hpp"
#include "State.hpp"

namespace limoncello {


States filter(const States& states, const double& start, const double& end) {

  States::reverse_iterator begin_prop_it;
	States::reverse_iterator end_prop_it;

  auto prop_it = states.begin();

	auto last_prop_it = prop_it;
	prop_it++;

	while (prop_it != states.end() && prop_it->stamp >= end) {
		last_prop_it = prop_it;
		prop_it++;
	}

	while (prop_it != states.end() && prop_it->stamp >= start) {
		prop_it++;
	}

	if (prop_it == states.end()) {
		return States();
	}

	prop_it++;

	end_prop_it = States::reverse_iterator(last_prop_it);
	begin_prop_it = States::reverse_iterator(prop_it);

  States out;
  for (auto it = begin_prop_it; it != end_prop_it; it++)
		out.push_back(*it);

  return out;
}


PointCloudT::Ptr deskew(const PointCloudT::Ptr& cloud,
                        const State& state,
                        const States& buffer,
												const double& offset) {
	
	auto binary_search = [&](const double& t) {
		int low = 0;
		int high = buffer.size() - 1;
		
		while (high >= low) {
			int mid = (low + high) / 2;
			if (buffer[mid].stamp > t)
				high = mid - 1;
			else
				low = mid + 1;
		}

		if (high < 0)
			return 0;

		return high;
	};

  PointCloudT::Ptr out(boost::make_shared<PointCloudT>());
  out->points.resize(cloud->points.size());

  std::vector<int> indices(cloud->points.size());
  std::iota(indices.begin(), indices.end(), 0);

	std::for_each(
		std::execution::par,
		indices.begin(),
		indices.end(),
		[&](int k) {
			int i_f = binary_search(cloud->points[k].stamp + offset);

			State X0 = buffer[i_f];
			X0.update(cloud->points[k].stamp + offset);

			Eigen::Affine3f T0 = X0.affine3f() * X0.I2L;
			Eigen::Affine3f TN = state.affine3f() * state.I2L;


			Eigen::Vector3f p << cloud->points[k].x,
													 cloud->points[k].y,
													 cloud->points[k].z;

			p = TN.inverse() * T0 * p;

			PointT pt;
			pt.x = p.x();
			pt.y = p.y();
			pt.z = p.z();
			pt.intensity = cloud->points[k].intensity;

			out->points[k] = pt;
		}
	);

	return out;
}


PointCloudT::Ptr process(const PointCloudT::Ptr& cloud) {
	Config& cfg = Config::getInstance();

	PointCloudT::Ptr out(boost::make_shared<PointCloudT>());

	int index = 0;
	std::copy_if(
		cloud->points.begin(), 
		cloud->points.end(), 
		std::back_inserter(out->points), 
		[&](const PointT& p) mutable {
				bool pass = true;

				// Distance filter
				if (cfg.filters.dist_active) {
						if (Eigen::Vector3f(p.x, p.y, p.z).norm() <= cfg.filters.min_dist)
								pass = false;
				}

				// Rate filter
				if (cfg.filters.rate_active) {
						if (index % cfg.filters.rate_value != 0)
								pass = false;
				}

				// Field of view filter
				if (cfg.filters.fov_active) {
						if (fabs(atan2(p.y, p.x)) >= cfg.filters.fov_angle)
								pass = false;
				}

				++index; // Increment index

				return pass;
		}
	);

	return out;
}


PointCloudT::Ptr downsample(const PointCloudT::Ptr& cloud) {
	Config& cfg = Config:::getInstance();

	static pcl::VoxelGrid<PointType> voxel_filter;
	voxel_filter.setLeafSize(Eigen::vector3f(cfg.filters.voxel.leafSize[0],
	                                          cfg.filters.voxel.leafSize[1],
																						cfg.filters.voxel.leafSize[2],
																						1.));

	PointCloudT::Ptr out(boost::make_shared<PointCloudT>());

	if (cfg.filters.voxel.active) {
		voxel_filter.setInputCloud(cloud);
		voxel_filter.filter(*out);
	} else {
		*out = *cloud;
	}

	return out;
}

}