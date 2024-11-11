/* 
 * This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
 * The code was obtained under the MIT license, which is distributed with this software
 * Copyright 2024 Fraunhofer IIS
*/ 
/* 
 * Implementation of the Wall class
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

#include <iostream>
#include <cmath>

#include "wall.hpp"
#include "geometry.hpp"
#include "common.hpp"
// M_PI is not defined in C++ standard in windows
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -->new for line 35 and 36
template<size_t D>
using Vectorf = Eigen::Matrix<float, D, 1>;


template<>
float Wall_deism<2>::area() const
{
  return (corners.col(1) - corners.col(0)).norm();
}

template<>
float Wall_deism<3>::area() const
{
  return area_2d_polygon(flat_corners);
}

// -->new line 52 to line 136, the function orderPoints is implemented in C++
// which will be used during the initialization of class "Wall"
template<>
Eigen::Matrix<float,2,Eigen::Dynamic> Wall_deism<2>::orderPoints(
  const Eigen::Matrix<float,2,Eigen::Dynamic>& points) {
    // template specialized for 2 dimensions
    // compute center of mass
    Vectorf<2> center=points.rowwise().mean();
    // choose reference vector
    Vectorf<2> refVector=points.col(0)-center;
    std::vector<std::pair<float,Vectorf<2>>> anglesWithPoints;

    for(int i=0;i<points.cols();i++){
      Vectorf<2> vecFromCenter=points.col(i)-center;
      float dotProduct=vecFromCenter.dot(refVector);
      float crossProduct=vecFromCenter.x()*refVector.y()-
        vecFromCenter.y()*refVector.x();
      float angle=std::atan2(crossProduct,dotProduct);
      if(angle<0){
        angle+=2*M_PI;
      }

      anglesWithPoints.emplace_back(angle,points.col(i));
    }

    std::sort(anglesWithPoints.begin(),anglesWithPoints.end(),
      [](const std::pair<float, Vectorf<2>>& a, 
        const std::pair<float, Vectorf<2>>& b){
          return a.first<b.first;
        }
    );

    Eigen::Matrix<float,2,Eigen::Dynamic> sortedPoints(2,points.cols());
    for(int i=0;i<points.cols();i++){
      sortedPoints.col(i)=anglesWithPoints[i].second;
    }

    return sortedPoints;
}

template<>
Eigen::Matrix<float,3,Eigen::Dynamic> Wall_deism<3>::orderPoints(
  const Eigen::Matrix<float,3,Eigen::Dynamic>& points) {
    // template specialized for 3 dimensions
    // compute center of mass
    Vectorf<3> center=points.rowwise().mean();
    // choose reference vector
    Vectorf<3> refVector=points.col(0)-center;
    std::vector<std::pair<float,Vectorf<3>>> anglesWithPoints;
    
    for(int i=0;i<points.cols();i++){
      Vectorf<3> vecFromCenter=points.col(i)-center;
      float dotProduct=vecFromCenter.dot(refVector);

      // calculate cross product in 3D for angle computation
      Vectorf<3> crossProduct=vecFromCenter.cross(refVector);
      float crossProductMagnitude = crossProduct.norm();
      float angle=std::atan2(crossProductMagnitude,dotProduct);

      if(normal.dot(crossProduct)<=0){
        angle=2*M_PI-angle;
      }

      anglesWithPoints.emplace_back(angle,points.col(i));
    }

    std::sort(anglesWithPoints.begin(),anglesWithPoints.end(),
      [](const std::pair<float, Vectorf<3>>& a, 
        const std::pair<float, Vectorf<3>>& b){
          return a.first<b.first;
        }
    );

    Eigen::Matrix<float,3,Eigen::Dynamic> sortedPoints(3,points.cols());
    for(int i=0;i<points.cols();i++){
      sortedPoints.col(i)=anglesWithPoints[i].second;
    }

    return sortedPoints;
}

template<size_t D>
Eigen::Matrix<float,D,Eigen::Dynamic> Wall_deism<D>::orderPoints(
  const Eigen::Matrix<float,D,Eigen::Dynamic>& points){
    return points;
}
// -->new

template<size_t D>
float Wall_deism<D>::area() const
{
  return 0.;
}

template<size_t D>
void Wall_deism<D>::init()
{
  // compute transmission coefficients from absorption
  energy_reflection.resize(absorption.size());
  energy_reflection = 1.f - absorption;
  transmission.resize(absorption.size());
  transmission = energy_reflection.sqrt();

  if (absorption.size() != scatter.size())
  {
    throw std::runtime_error("The number of absorption and scattering coefficients is different");
  }
}

// original constructor without centroid and impedance
template<>
Wall_deism<2>::Wall_deism(
    const Eigen::Matrix<float,2,Eigen::Dynamic> &_corners,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name
    )
  : absorption(_absorption), scatter(_scatter), name(_name), corners(_corners)
{
  /*
  initialize 2D walls, the first dot in _corners is the original of wall.
  */
  init();

  // Pick one of the corners as the origin of the wall
  origin = corners.col(0);

  // compute normal (difference of 2 corners, swap x-y, change 1 sign)
  normal.coeffRef(0) = corners.coeff(1,1) - corners.coeff(1,0);
  normal.coeffRef(1) = corners.coeff(0,0) - corners.coeff(0,1);
  // 
  normal = normal.normalized();

  /****/
  // after computing norm, then reflection matrix is also computed
  // first set as a DxD Identity
  reflection_matrix.setIdentity();  //-->new
  // calculate reflection matrix
  reflection_matrix-=2*normal*normal.transpose();   //-->new
}

template<>
Wall_deism<2>::Wall_deism(
    const Eigen::Matrix<float,2,Eigen::Dynamic> &_corners,
    const Eigen::Matrix<float,2,1>& _centroid,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name
    )
  : absorption(_absorption), scatter(_scatter), name(_name), corners(_corners),
    centroid(_centroid)
{
  /*
  initialize 2D walls, the first dot in _corners is the original of wall.
  */
  init();

  // Pick one of the corners as the origin of the wall
  origin = corners.col(0);

  // compute normal (difference of 2 corners, swap x-y, change 1 sign)
  normal.coeffRef(0) = corners.coeff(1,1) - corners.coeff(1,0);
  normal.coeffRef(1) = corners.coeff(0,0) - corners.coeff(0,1);
  // 
  normal = normal.normalized();

  /****/
  // after computing norm, then reflection matrix is also computed
  // first set as a DxD Identity
  reflection_matrix.setIdentity();  //-->new
  // calculate reflection matrix
  reflection_matrix-=2*normal*normal.transpose();   //-->new
}

/******************************************************************************/
// new constructors
template<>
Wall_deism<2>::Wall_deism(
    const Eigen::Matrix<float,2,Eigen::Dynamic> &_corners,
    const Eigen::Matrix<float,2,1>& _centroid,
    float _impedance,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name
  )
  : Wall_deism<2>::Wall_deism(_corners,_centroid,_absorption,_scatter,_name) 
{
  impedance=_impedance;
}
/******************************************************************************/


// -->original Wall_deism<3>
template<>
Wall_deism<3>::Wall_deism(
    const Eigen::Matrix<float,3,Eigen::Dynamic> &_corners,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name
    )
  : absorption(_absorption), scatter(_scatter), name(_name), corners(_corners)
{
  init();

  // In 3D things are a little more complicated
  // We need to compute a 2D basis for the plane and find the normal

  // Pick the origin as the first corner
  origin = corners.col(0);

  // order points in counter-clockwise
  Eigen::Matrix<float,3,Eigen::Dynamic> ordered_points=orderPoints(corners);
  // The basis and normal are found by SVD
  Eigen::JacobiSVD<Eigen::Matrix<float,3,Eigen::Dynamic>> svd(corners.colwise() - origin, Eigen::ComputeThinU);

  // The corners matrix should be rank defficient, check the smallest eigen value
  // The rank deficiency is because all the corners are in a 2D subspace of 3D space
  if (svd.singularValues().coeff(2) > libroom_eps)
  {
    throw std::runtime_error("The corners of the wall do not lie in a plane");
  }

  // The basis is the leading two left singular vectors
  basis.col(0) = svd.matrixU().col(0);
  basis.col(1) = svd.matrixU().col(1);

  // The normal corresponds to the smallest singular value
  normal = svd.matrixU().col(2);

  // Project the 3d corners into 2d plane
  flat_corners = basis.adjoint() * (corners.colwise() - origin);

  // Our convention is that the vertices are arranged counter-clockwise
  // around the normal. In that case, the area computation should be positive.
  // If it is positive, we need to swap the basis.
  float a = area();
  if (a < 0)
  {
    // exchange the other two basis vectors
    basis.rowwise().reverseInPlace();
    flat_corners.colwise().reverseInPlace();
  }


  // Now the normal is computed as the cross product of the two basis vectors
  normal = cross(basis.col(0), basis.col(1));

  reflection_matrix.setIdentity();  //-->new
  // calculate reflection matrix
  reflection_matrix-=2*normal*normal.transpose();   //-->new
}


// -->new Wall_deism<3>
template<>
Wall_deism<3>::Wall_deism(
    const Eigen::Matrix<float,3,Eigen::Dynamic> &_corners,
    const Eigen::Matrix<float,3,1>& _centroid,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name    
    )
  : absorption(_absorption), scatter(_scatter), name(_name), corners(_corners),
    centroid(_centroid)
{
  init();

  Eigen::Matrix<float,3,1> v1=corners.col(1)-corners.col(0);
  Eigen::Matrix<float,3,1> v2=corners.col(2)-corners.col(0);
  normal=v1.cross(v2).normalized();

  // In 3D things are a little more complicated
  // We need to compute a 2D basis for the plane and find the normal

  // Pick the origin as the first corner
  origin = corners.col(0);

  if(normal.dot(centroid-origin)>0){
    normal=-normal;
  }

  corners=orderPoints(corners);
  // -->new, the default behavior of SVD function in eigen is not completely the same 
  // with that in python, which will cause, in the same surface, the length of the 
  // basis is the same, but the direction may be inverted, As a result, the 
  // coordinates of projected points(flat_corners) may be different in signs.
  
  // The basis and normal are found by SVD
  Eigen::JacobiSVD<Eigen::Matrix<float,3,Eigen::Dynamic>> svd(
    corners.colwise() - origin,Eigen::ComputeThinU | Eigen::ComputeThinV);

  // The corners matrix should be rank defficient, check the smallest eigen value
  // The rank deficiency is because all the corners are in a 2D subspace of 3D space
  if (svd.singularValues().coeff(2) > libroom_eps)
  {
    throw std::runtime_error("The corners of the wall do not lie in a plane");
  }

  // The basis is the leading two left singular vectors
  basis.col(0) = svd.matrixU().col(0);
  // to make sure the results are the same with svd in python, but the direction of 
  // basis vector has no more meanings
  basis.col(1) = -1*svd.matrixU().col(1);

  // basis = svd.matrixU().leftCols(2);
  flat_corners=basis.transpose()*(corners.colwise()-origin);
  float a=area();
  if(a<0){
    basis.rowwise().reverseInPlace();
    flat_corners.colwise().reverseInPlace();
  }

  reflection_matrix.setIdentity();
  reflection_matrix-=2*normal*normal.transpose();
}


/******************************************************************************/
// new 3d walls constructor
template<>
Wall_deism<3>::Wall_deism(
    const Eigen::Matrix<float,3,Eigen::Dynamic> &_corners,
    const Eigen::Matrix<float,3,1>& _centroid,
    float _impedance,
    const Eigen::ArrayXf &_absorption,
    const Eigen::ArrayXf &_scatter,
    const std::string &_name
  ):Wall_deism<3>::Wall_deism(_corners,_centroid,_absorption,_scatter,_name)
{
    impedance=_impedance;
}
/******************************************************************************/

template<>
int Wall_deism<2>::intersection(
    const Eigen::Matrix<float,2,1> &p1,
    const Eigen::Matrix<float,2,1> &p2,
    Eigen::Ref<Eigen::Matrix<float,2,1>> intersection
    ) const
{
  return intersection_2d_segments(p1, p2, corners.col(0), corners.col(1), intersection);
}

template<>
int Wall_deism<3>::intersection(
    const Eigen::Matrix<float,3,1> &p1,
    const Eigen::Matrix<float,3,1> &p2,
    Eigen::Ref<Eigen::Matrix<float,3,1>> intersection
    ) const
{
  /*
    Computes the intersection between a line segment and a polygon surface in 3D.
    This function computes the intersection between a line segment (defined
    by the coordinates of two points) and a surface (defined by an array of
    coordinates of corners of the polygon and a normal vector)
    If there is no intersection, None is returned.
    If the segment belongs to the surface, None is returned.
    Two booleans are also returned to indicate if the intersection
    happened at extremities of the segment or at a border of the polygon,
    which can be useful for limit cases computations.

    a1: (array size 3) coordinates of the first endpoint of the segment
    a2: (array size 3) coordinates of the second endpoint of the segment
    corners: (array size 3xN, N>2) coordinates of the corners of the polygon
    normal: (array size 3) normal vector of the surface
    intersection: (array size 3) store the intersection point

    :returns: 
           -1 if there is no intersection
            0 if the intersection striclty between the segment endpoints and in the polygon interior
            1 if the intersection is at endpoint of segment
            2 if the intersection is at boundary of polygon
            3 if both the above are true
    */

  int ret1, ret2, ret = 0;

  ret1 = intersection_3d_segment_plane(p1, p2, origin, normal, intersection);

  if (ret1 == -1)
    return -1;  // there is no intersection

  if (ret1 == 1)  // intersection at endpoint of segment
    ret = 1;

  /* project intersection into plane basis */
  Eigen::Vector2f flat_intersection = basis.adjoint() * (intersection - origin);

  /* check in flatland if intersection is in the polygon */
  ret2 = is_inside_2d_polygon(flat_intersection, flat_corners);

  if (ret2 < 0)  // intersection is outside of the wall
    return -1;

  if (ret2 == 1) // intersection is on the boundary of the wall
    ret |= 2;

  return ret;  // no intersection
}



template<size_t D>
int Wall_deism<D>::intersects(const Vectorf<D> &p1, const Vectorf<D> &p2) const
{
  Vectorf<D> v;
  return intersection(p1, p2, v);
}

template<size_t D>
int Wall_deism<D>::reflect(const Vectorf<D> &p, Eigen::Ref<Vectorf<D>> p_reflected) const
{
  /*
   * Reflects point p across the wall 
   *
   * wall: a wall object (2d or 3d)
   * p: a point in space
   * p_reflected: a pointer to a buffer large enough to receive
   *              the location of the reflected point
   *
   * Returns: 1 if reflection is in the same direction as the normal
   *          0 if the point is within tolerance of the wall
   *         -1 if the reflection is in the opposite direction of the normal
   */

  // projection onto normal axis
  float distance_wall2p = normal.adjoint() * (origin - p);

  // compute reflected point
  p_reflected = p + 2 * distance_wall2p * normal;

  if (distance_wall2p > libroom_eps)
    return 1;
  else if (distance_wall2p < -libroom_eps)
    return -1;
  else
    return 0;
}


/* checks on which side of a wall a point is */
template<size_t D>
int Wall_deism<D>::side(const Vectorf<D> &p) const
{
  // Essentially, returns the sign of the inner product with the normal vector
  float ip = (p - origin).adjoint() * normal;

  if (ip > libroom_eps)
    return 1;
  else if (ip < -libroom_eps)
    return -1;
  else
    return 0;
}

template<size_t D>
bool Wall_deism<D>::same_as(const Wall_deism & that) const
{
  /*
  Checks if two walls are the same, based on their corners of the walls.
  Be careful : it will return true for two identical walls that belongs
  to two different rooms !
  */

  if (dim != that.dim)
  {
    std::cerr << "The two walls are not of the same dimensions !" << std::endl;
    return false;
  }

  // Not the same number of corners
  if (corners.cols() != that.corners.cols())
  {
    return false;
  }

  return (corners - that.corners).cwiseAbs().sum() == 0.;
}

template<size_t D>
Vectorf<D> Wall_deism<D>::normal_reflect(
    const Vectorf<D> &start,
    const Vectorf<D> &hit_point,
    float length) const
{
	  
  /* This method computes the reflection of one point with respect to
   a precise hit_point on a wall. Also, the distance between the
   wall hit point and the reflected point is defined by the 'length'
   parameter.
   This method computes the reflection of point 'start' across the normal
   to the wall through 'hit_point'.
    
   start: (array size 2 or 3) defines the point to be reflected
   hit_point: (array size 2 or 3) defines a point on a wall that will
     serve as the reference point for the reflection
   wall_normal: (array size 2 or 3) defines the normal of the reflecting
     wall. It will be used as if it was anchored at hit_point
   length : the desired distance between hit_point and the reflected point
   
   :returns: an array of size 2 or 3 representing the reflected point
   */

  Vectorf<D> incident = (hit_point - start).normalized();
  // return hit_point + length * normal_reflect(incident);
  return hit_point + length * (incident - normal * 2 * incident.dot(normal));
}

template<size_t D>
Vectorf<D> Wall_deism<D>::normal_reflect(const Vectorf<D> &incident) const
{
  /*
   * Same as the previous function, but works on a direction vector instead
   */
  return incident - normal * 2 * incident.dot(normal);
}

template<size_t D>
float Wall_deism<D>::cosine_angle(
    const Vectorf<D> &p) const
{
    /*
    Compute the cosine angle between the surface normal and a given vector.
    */

    return p.dot(normal) / p.norm();

}

/**************************************************************************/
// area for new member functions
template<size_t D>
float Wall_deism<D>::get_attenuation(float theta) const {
    return (this->impedance*cos(theta)-1)/(this->impedance*cos(theta)+1);
};

/**************************************************************************/



