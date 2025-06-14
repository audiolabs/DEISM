/* 
 * This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
 * The code was obtained under the MIT license, which is distributed with this software
 * Copyright 2024 Fraunhofer IIS
*/ 
/* 
 * Definition of the Room_deism class
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program. If
 * not, see <https://opensource.org/licenses/MIT>.
 */
#ifndef __ROOM_H__
#define __ROOM_H__

#include <vector>
#include <stack>
#include <tuple>
#include <Eigen/Dense>
#include <algorithm>
#include <ctime>

#include "common.hpp"
#include "wall.hpp"

#include "microphone.hpp"   //-->new

template<size_t D>
struct ImageSource
{
	/*
	 * A class to hold the information relating to an Image source when running the ISM
	 */

	Vectorf<D> loc;
	Eigen::ArrayXf attenuation;
	int order;
	int gen_wall;
	ImageSource *parent;
	VectorXb visible_mics;

	// this is a unit vector from the center of the source pointing
	// in the direction of the path to the microphone
	Vectorf<D> source_impact_dir;

	// This contains the reflection orders with respect to x/y/z axis
	// for the shoebox image source model
	Vectori<D> order_xyz;

	/****************************************************************************/
	// adding a DxD identity matrix
	Eigen::Matrix<float,D,D> reflection_matrix;
	// std::vector<Vectorf<D>>& list_intercep_p_to_is={};
	/****************************************************************************/

	ImageSource(size_t n_bands)
		: order(0), gen_wall(-1), parent(NULL)
	{
		loc.setZero();
		attenuation.resize(n_bands);
		attenuation.setOnes();

		/**************************************************************************/
		reflection_matrix.setIdentity();
		/**************************************************************************/
	}

	ImageSource(const Vectorf<D> &_loc, size_t n_bands)
		: loc(_loc), order(0), gen_wall(-1), parent(NULL)
	{
		attenuation.resize(n_bands);
		attenuation.setOnes();

		/**************************************************************************/
		reflection_matrix.setIdentity();
		/**************************************************************************/
	}



	/****************************************************************************/
	// implement a copy instructor
	ImageSource(const ImageSource<D>& other)
		: loc(other.loc),
			attenuation(other.attenuation),
			order(other.order),
			gen_wall(other.gen_wall),
			parent(other.parent),
			visible_mics(other.visible_mics),
			source_impact_dir(other.source_impact_dir),
			order_xyz(other.order_xyz),
			reflection_matrix(other.reflection_matrix)
	{

	}
	/****************************************************************************/



};

/*
 * Structure for a room as a list of walls
 * with a few sources and microphones around
 */
template<size_t D>
class Room_deism
{
public:
	static const int dim = D;

	std::vector<Wall_deism<D>> walls;
	std::vector<int> obstructing_walls;  // List of obstructing walls
	std::vector<Microphone_deism<D>> microphones;  // The microphones are in the room
	float sound_speed = 343.;  // the speed of sound in the room

	// Simulation parameters
	int ism_order = 0.;

	// Ray tracing parameters
	float energy_thres = 1e-7;
	float time_thres = 1.;
	float mic_radius = 0.15f;  // receiver radius in meters
	double mic_radius_sq = 0.15f * 0.15f;  // receiver radius in meters
	float mic_hist_res = 0.004;  // in seconds
	bool is_hybrid_sim = true;

	// Special parameters for shoebox rooms
	bool is_shoebox = false;
	Vectorf<D> shoebox_size;
	Eigen::Array<float,Eigen::Dynamic,2*D> shoebox_absorption;
	Eigen::Array<float,Eigen::Dynamic,2*D> shoebox_scattering;

	// The number of frequency bands used
	size_t n_bands;
	// 2. A distance after which a ray must have hit at least 1 wall
	float max_dist = 0.;

	// This is a list of image sources
	Eigen::Matrix<float,D,Eigen::Dynamic> sources;
	Eigen::VectorXi gen_walls;
	Eigen::VectorXi orders;
	Eigen::Matrix<int, D, Eigen::Dynamic> orders_xyz;
	Eigen::MatrixXf attenuations;

	// This array will get filled by visibility status
	// its size is n_microphones * n_sources
	MatrixXb visible_mics;



	// area for new parameters 
	/*20240705*/
	/**************************************************************************/
	/*                                                                        */
	// impedence is a ndarray value, each value correspondes to a wall, its value 
	// type is complex, like[], but now is only a integer number
	// float impedence;   // set initial value of impedence

	// the impedence is connected with frequency, then it should be defined as a 
	// dynamic array
	Eigen::ArrayXf impedence_bands;
	// the last dimension of reflection matrix is dynamic, which depends on running
	std::vector<Eigen::Matrix<float,D,D>> reflection_matrix;
	/*                                                                        */
	/**************************************************************************/
	


	// Constructor for general rooms
	Room_deism(
		const std::vector<Wall_deism<D>> &_walls,
		const std::vector<int> &_obstructing_walls,
		const std::vector<Microphone_deism<D>> &_microphones,
		float _sound_speed,
		// parameters for the image source model
		int _ism_order,
		// parameters for the ray tracing
		float _energy_thres,
		float _time_thres,
		float _mic_radius,
		float _mic_hist_res,
		bool _is_hybrid_sim
	);

	// Constructor for shoebox rooms
	Room_deism(
		const Vectorf<D> &_room_size,
		const Eigen::Array<float,Eigen::Dynamic,2*D> &_absorption,
		const Eigen::Array<float,Eigen::Dynamic,2*D> &_scattering,
		const std::vector<Microphone_deism<D>> &_microphones,
		float _sound_speed,
		// parameters for the image source model
		int _ism_order,
		// parameters for the ray tracing
		float _energy_thres,
		float _time_thres,
		float _mic_radius,
		float _mic_hist_res,
		bool _is_hybrid_sim
	);



	/**************************************************************************/
	// new constructor after increasing impedence variable
	// Constructor for general rooms
	// this is the main constructor we need
	// Room_deism(
	// 	const std::vector<Wall_deism<D>> &_walls,
	// 	const std::vector<int> &_obstructing_walls,
	// 	const std::vector<Microphone_deism<D>> &_microphones,
	// 	float _sound_speed,
	// 	// parameters for the image source model
	// 	int _ism_order,
	// 	// parameters for the ray tracing
	// 	float _energy_thres,
	// 	float _time_thres,
	// 	float _mic_radius,
	// 	float _mic_hist_res,
	// 	bool _is_hybrid_sim
	// );

	// // Constructor for shoebox rooms
	// Room_deism(
	// 	const Vectorf<D> &_room_size,
	// 	const Eigen::Array<float,Eigen::Dynamic,2*D> &_absorption,
	// 	const Eigen::Array<float,Eigen::Dynamic,2*D> &_scattering,
	// 	const std::vector<Microphone_deism<D>> &_microphones,
	// 	float _sound_speed,
	// 	// parameters for the image source model
	// 	int _ism_order,
	// 	// parameters for the ray tracing
	// 	float _energy_thres,
	// 	float _time_thres,
	// 	float _mic_radius,
	// 	float _mic_hist_res,
	// 	bool _is_hybrid_sim
	// );
	/**************************************************************************/



	void make_shoebox_walls(
		const Vectorf<D> &rs,  // room_size
		const Eigen::Array<float,Eigen::Dynamic,2*D> &abs,
		const Eigen::Array<float,Eigen::Dynamic,2*D> &scat
	);

	void init();

	void set_params(
		float _sound_speed,
		int _ism_order,
		float _energy_thres,
		float _time_thres,
		float _mic_radius,
		float _mic_hist_res,
		bool _is_hybrid_sim
	)
	{
		sound_speed = _sound_speed;
		ism_order = _ism_order;
		energy_thres = _energy_thres;
		time_thres = _time_thres;
		mic_radius = _mic_radius;
		mic_radius_sq = _mic_radius * _mic_radius;
		mic_hist_res = _mic_hist_res;
		is_hybrid_sim = _is_hybrid_sim;
	}



	/**************************************************************************/
	// overloaded set_params
	// void set_params(
	// 	float _sound_speed,
	// 	int _ism_order,
	// 	float _energy_thres,
	// 	float _time_thres,
	// 	float _mic_radius,
	// 	float _mic_hist_res,
	// 	bool _is_hybrid_sim,
	// 	float _impedence
	// )
	// {
	// 	sound_speed = _sound_speed;
	// 	ism_order = _ism_order;
	// 	energy_thres = _energy_thres;
	// 	time_thres = _time_thres;
	// 	mic_radius = _mic_radius;
	// 	mic_radius_sq = _mic_radius * _mic_radius;
	// 	mic_hist_res = _mic_hist_res;
	// 	is_hybrid_sim = _is_hybrid_sim;
	// 	impedence=_impedence;
	// }

	Eigen::ArrayXf get_image_attenuation(ImageSource<D>& old_is,
										std::vector<Vectorf<D>>& list_intercep_p_to_is);

	// define member functions that modifies n_bands
	void set_n_bands(int _n_bands) { n_bands = _n_bands; }
	int get_n_bands() { return n_bands; }

	/**************************************************************************/



	void set_is_hybrid_sim(bool state) { is_hybrid_sim = state; }
	bool get_is_hybrid_sim() { return is_hybrid_sim; }

	void add_mic(const Vectorf<D> &loc)
	{
		microphones.push_back(
			Microphone_deism<D>(loc, n_bands, mic_hist_res * sound_speed, 
				time_thres * sound_speed)
		);
	}

	void reset_mics()
	{
		for (auto mic = microphones.begin() ; mic != microphones.end() ; ++mic)
			mic->reset();
	}

	Wall_deism<D> & get_wall(int w) { return walls[w]; }

	// Image source model methods
	int image_source_model(const Vectorf<D> &source_location);

	float get_max_distance();

	std::tuple < Vectorf<D>, int, float > next_wall_hit(
		const Vectorf<D> &start,
		const Vectorf<D> &end,
		bool scattered_ray
	);

	bool scat_ray(
		const Eigen::ArrayXf &transmitted,
		const Wall_deism<D> &wall,
		const Vectorf<D> &prev_last_hit,
		const Vectorf<D> &hit_point,
		float travel_dist
	);

	void simul_ray(
		float phi,
		float theta,
		const Vectorf<D> source_pos,
		float energy_0
	);

	void ray_tracing(
		const Eigen::Matrix<float,D-1,Eigen::Dynamic> &angles,
		const Vectorf<D> source_pos
	);

	void ray_tracing(
		size_t nb_phis,
		size_t nb_thetas,
		const Vectorf<D> source_pos
	);

	void ray_tracing(
		size_t n_rays,
		const Vectorf<D> source_pos
	);

	bool contains(const Vectorf<D> point);

	// if we need to export the following methods to python, then these methods 
	// should not be private
// private:   //-->new
	// We need a stack to store the image sources during the algorithm
	std::stack<ImageSource<D>> visible_sources;

	// A specialized method for the shoebox room case
	int image_source_shoebox(const Vectorf<D> &source);

	// Image source model internal methods
	void image_sources_dfs(ImageSource<D> &is, int max_order);
	std::pair<bool,std::vector<Vectorf<D>>> is_visible_dfs(const Vectorf<D> &p, 
											ImageSource<D> &is);
	bool is_obstructed_dfs(const Vectorf<D> &p, ImageSource<D> &is);
	int fill_sources();

};


#include "room.cpp"

#endif // __ROOM_H__
