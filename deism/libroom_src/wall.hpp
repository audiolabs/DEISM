/* 
 * This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
 * The code was obtained under the MIT license, which is distributed with this software
 * Copyright 2024 Fraunhofer IIS
*/ 
/* 
 * Definition of the Wall class used in libroom core of pyroomacoustics
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
#ifndef __CWALL_H__
#define __CWALL_H__

#include <string>
#include <Eigen/Dense>
#include <vector>       //-->new

// -->new for line 33 and 34
template<size_t D>
using Vectorf = Eigen::Matrix<float, D, 1>;

float libroom_eps=1e-5;

#define WALL_ISECT_NONE        -1  // if there is no intersection
#define WALL_ISECT_VALID        0  // if the intersection striclty between the segment endpoints and in the polygon interior
#define WALL_ISECT_VALID_ENDPT  1  // if the intersection is at endpoint of segment
#define WALL_ISECT_VALID_BNDRY  2  // if the intersection is at boundary of polygon
#define ENDPOINT_BOUNDARY       3  // if both the above are true

template<size_t D>
class Wall_deism
{
  private:
    void init();  // common part of initialization for walls of any dimension

  public:
    enum Isect {  // The different cases for intersections
      NONE = -1,  // - There is no intersection
      VALID = 0,  // - There is a valid intersection
      ENDPT = 1,  // - The intersection is on the endpoint of the segment
      BNDRY = 2   // - The intersection is on the boundary of the wall
    };

    static const int dim = D;

    // Wall_deism properties container
    Eigen::ArrayXf absorption;  // the wall absorption coefficient for every freq. band
    Eigen::ArrayXf scatter;  // the wall scattering coefficient for every freq. band
    std::string name;
    Eigen::ArrayXf transmission;  // computed from absorption as sqrt(1 - a)
    Eigen::ArrayXf energy_reflection;  // computed from absorption as (1 - a)
    
    // Wall_deism geometry properties
    Eigen::Matrix<float, D, 1>  normal;
    Eigen::Matrix<float, D, Eigen::Dynamic> corners;

    /* for 3D wall, provide local basis for plane of wall */
    Eigen::Matrix<float, D, 1> origin;
    Eigen::Matrix<float, D, 2> basis;
    Eigen::Matrix<float, 2, Eigen::Dynamic> flat_corners;

    /**************************************************************************/
    // reflection matrix
    Eigen::Matrix<float,D,D> reflection_matrix;     //-->new
    // centroid means the centroid of the room, used to redirect the direction of 
    // norm vectors of walls
    Eigen::Matrix<float,D,1> centroid;    //-->new
    /**************************************************************************/


    /**************************************************************************/
    // area for new parameters
    // float impedence;        //-->new

    // if impedence is connected with frequency, then how to set the corresponding 
    // impedence under some freqneucy? use map? the question is how do we get 
    // the impedence of e.g. 1000Hz? since impedence is defined as a dynamic array
    // n_bands-->how many frequency bands are there.
    // int n_bands;    //-->new
    Eigen::ArrayXf impedence_bands;     //-->new
    // solution: define impedence under certain points and propagate to all 
    // frequency by interpolation
    /**************************************************************************/

    // Constructor
    Wall_deism(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        const Eigen::ArrayXf &_absorption,
        const Eigen::ArrayXf &_scatter,
        const std::string &_name
        );
    Wall_deism(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        const Eigen::ArrayXf &_absorption,
        const Eigen::ArrayXf &_scatter
        ) : Wall_deism(_corners, _absorption, _scatter, "") {}

    /**************************************************************************/
    // area for new constructors
    // inserting centroid
    Wall_deism(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        const Eigen::Matrix<float,D,1>& centroid,  //-->new
        const Eigen::ArrayXf &_absorption,
        const Eigen::ArrayXf &_scatter,
        const std::string &_name
        );
    Wall_deism(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        const Eigen::Matrix<float,D,1>& _centroid,  //-->new
        const Eigen::ArrayXf &_absorption,
        const Eigen::ArrayXf &_scatter
        ) : Wall_deism(_corners,_centroid, _absorption, _scatter, "") {}


    // Wall_deism(
    //     const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
    //     const Eigen::Matrix<float,D,1>& _centroid,    //-->new
    //     float _impedence,
    //     const Eigen::ArrayXf &_absorption,
    //     const Eigen::ArrayXf &_scatter,
    //     const std::string &_name
    //   );

    // the main constructor we should consider
    Wall_deism(
        const Eigen::Matrix<float, D, Eigen::Dynamic> &_corners,
        const Eigen::Matrix<float,D,1>& _centroid,    //-->new
        const Eigen::ArrayXf &_impedence_bands,         //-->new
        const Eigen::ArrayXf &_absorption,
        const Eigen::ArrayXf &_scatter,
        const std::string &_name
    );

    /**************************************************************************/

    // Copy constructor
    // here only copy parameter impedence_bands, but not consider the parameter impedence(float)
    Wall_deism(const Wall_deism<D> &w) :
      absorption(w.absorption), scatter(w.scatter), name(w.name),
      transmission(w.transmission), energy_reflection(w.energy_reflection),
      normal(w.normal), corners(w.corners),
      origin(w.origin), basis(w.basis), flat_corners(w.flat_corners),
      impedence_bands(w.impedence_bands),reflection_matrix(w.reflection_matrix),
      centroid(w.centroid)
    {}

    // public methods
    const Eigen::ArrayXf &get_transmission() const { return transmission; }
    const Eigen::ArrayXf &get_energy_reflection() const { return energy_reflection; }
    size_t get_n_bands() const { return transmission.size(); }
    float area() const;  // compute the area of the wall
    int intersection(  // compute the intersection of line segment (p1 <-> p2) with wall
        const Vectorf<D> &p1,
        const Vectorf<D> &p2,
        Eigen::Ref<Vectorf<D>> intersection
    ) const;
    /*-->new 
    for the above method, p1 and p2 should be two points,while intersection 
    is the coordinates of the intersected point with the wall?
    */

    int intersects(
        const Vectorf<D> &p1,
        const Vectorf<D> &p2
        ) const;

    int reflect(
        const Vectorf<D> &p,
        Eigen::Ref<Vectorf<D>> p_reflected
        ) const;
    int side(const Vectorf<D> &p) const;
    bool same_as(const Wall_deism & that) const;

    Vectorf<D> normal_reflect(
        const Vectorf<D> &start,
        const Vectorf<D> &hit_point,
        float length) const;

    Vectorf<D> normal_reflect(const Vectorf<D> &incident) const;

    float cosine_angle(   // cosine angle with respect to surface normal
        const Vectorf<D> &p
        ) const;

    /**************************************************************************/
    // area for new member functions
    // since impedence is a array, so the attenuation should be a array,too.
    // float get_attenuation(float theta) const;   //-->new
    Eigen::ArrayXf get_attenuation(float theta) const;   //-->new

    Eigen::Matrix<float,D,Eigen::Dynamic> orderPoints(
        const Eigen::Matrix<float,D,Eigen::Dynamic>& points);   //-->new

    /**************************************************************************/
}; 


#include "wall.cpp"

#endif // __CWALL_H__
