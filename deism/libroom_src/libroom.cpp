/* 
 * This function originated from https://github.com/LCAV/pyroomacoustics and is modified by Fraunhofer
 * The code was obtained under the MIT license, which is distributed with this software
 * Copyright 2024 Fraunhofer IIS
*/ 
/*
 * Python bindings for libroom
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program.
 * If not, see <https://opensource.org/licenses/MIT>.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

// extern float libroom_eps = 1e-5;  // epsilon is set to 0.1 millimeter (100 um)

#include "common.hpp"
#include "geometry.hpp"
#include "microphone.hpp"
#include "rir_builder.hpp"
#include "room.hpp"
#include "wall.hpp"

namespace py = pybind11;

// float libroom_eps = 1e-5;  // epsilon is set to 0.1 millimeter (100 um)

PYBIND11_MODULE(libroom_deism, m) {
  m.doc() =
      "Libroom room simulation extension plugin";  // optional module docstring

  // The 3D Room_deism class
  py::class_<Room_deism<3>>(m, "Room_deism")
        .def(py::init<const std::vector<Wall_deism<3>> &, const std::vector<int> &,
                    const std::vector<Microphone_deism<3>> &, float, int, float,
                    float, float, float, bool>())
        .def(py::init<const Vectorf<3> &,
                    const Eigen::Array<float, Eigen::Dynamic, 6> &,
                    const Eigen::Array<float, Eigen::Dynamic, 6> &,
                    const std::vector<Microphone_deism<3>> &, float, int, float,
                    float, float, float, bool>())
        .def(py::init<const std::vector<Wall_deism<3>> &,const std::vector<int> &,
                    const std::vector<Microphone_deism<3>> &, float, int, float,
                    float, float, float, bool, float>())    //tan new constructor←←
        .def(py::init<const Vectorf<3> &,
                    const Eigen::Array<float,Eigen::Dynamic,2*3> &,
                    const Eigen::Array<float,Eigen::Dynamic,2*3> &,
                    const std::vector<Microphone_deism<3>> &, float, int, float,
                    float, float, float, bool, float>())    //tan new constructor←←
    //   .def("set_params", &Room_deism<3>::set_params)
        .def("set_params",static_cast<void (Room_deism<3>::*)(float,int,float,float,
                    float,float,bool)>(&Room_deism<3>::set_params))   //original
        .def("set_params",static_cast<void (Room_deism<3>::*)(float,int,float,float,
                    float,float,bool,float)>(&Room_deism<3>::set_params))     //tan overloaded
        .def("get_image_attenuation",&Room_deism<3>::get_image_attenuation)   //tan new added
        .def("add_mic", &Room_deism<3>::add_mic)
        .def("reset_mics", &Room_deism<3>::reset_mics)
        .def("image_source_model", &Room_deism<3>::image_source_model)
        .def("get_wall", &Room_deism<3>::get_wall)
        .def("get_max_distance", &Room_deism<3>::get_max_distance)
        .def("next_wall_hit", &Room_deism<3>::next_wall_hit)
        .def("scat_ray", &Room_deism<3>::scat_ray)
        .def("simul_ray", &Room_deism<3>::simul_ray)
        .def("ray_tracing",
            (void(Room_deism<3>::*)(
                const Eigen::Matrix<float, 2, Eigen::Dynamic> &angles,
                const Vectorf<3> source_pos)) &
                Room_deism<3>::ray_tracing)
        .def("ray_tracing", (void(Room_deism<3>::*)(size_t nb_phis, size_t nb_thetas,
                                                const Vectorf<3> source_pos)) &
                                Room_deism<3>::ray_tracing)
        .def("ray_tracing",
            (void(Room_deism<3>::*)(size_t nb_rays, const Vectorf<3> source_pos)) &
                Room_deism<3>::ray_tracing)
        .def("contains", &Room_deism<3>::contains)
        .def("image_sources_dfs",(void(Room_deism<3>::*)(ImageSource<3> &is, int max_order))&Room_deism<3>::image_sources_dfs)      //-->new 
        // .def("image_sources_dfs",(void(Room_deism<3>::*)(ImageSource<3> &is, int max_order,
        //     std::vector<Vectorf<3>>& list_intercep_p_to_is))&Room_deism<3>::image_sources_dfs)    //-->new
        // .def("is_visible_dfs",(bool (Room_deism<3>::*)(const Vectorf<3> &p, ImageSource<3> &is))&Room_deism<3>::is_visible_dfs)    //-->new
        // .def("is_visible_dfs",(bool (Room_deism<3>::*)(const Vectorf<3> &p, ImageSource<3> &is,
        //     std::vector<Vectorf<3>>& list_intercep_p_to_is))&Room_deism<3>::is_visible_dfs)    //-->new
        .def("is_visible_dfs",(std::pair<bool,std::vector<Vectorf<3>>> (Room_deism<3>::*)(
            const Vectorf<3> &p, ImageSource<3> &is))&Room_deism<3>::is_visible_dfs)    //-->new
        .def("is_obstructed_dfs",&Room_deism<3>::is_obstructed_dfs)   //-->new
        .def("fill_sources",&Room_deism<3>::fill_sources)     //-->new
        .def_property("is_hybrid_sim", &Room_deism<3>::get_is_hybrid_sim,
                        &Room_deism<3>::set_is_hybrid_sim)
        .def_property_readonly_static("dim",
                                        [](py::object /* self */) { return 3; })
        .def_readonly("walls", &Room_deism<3>::walls)
        .def_readonly("sources", &Room_deism<3>::sources)
        .def_readonly("orders", &Room_deism<3>::orders)
        .def_readonly("orders_xyz", &Room_deism<3>::orders_xyz)
        .def_readonly("attenuations", &Room_deism<3>::attenuations)
        .def_readonly("gen_walls", &Room_deism<3>::gen_walls)
        .def_readonly("visible_mics", &Room_deism<3>::visible_mics)
        .def_readonly("walls", &Room_deism<3>::walls)
        .def_readonly("obstructing_walls", &Room_deism<3>::obstructing_walls)
        .def_readonly("microphones", &Room_deism<3>::microphones)
        .def_readonly("max_dist", &Room_deism<3>::max_dist)
        .def_readonly("impedance",&Room_deism<3>::impedance)  //--> new
        .def_readonly("reflection_matrix",&Room_deism<3>::reflection_matrix)  //--> new
        .def_readwrite("n_bands",&Room_deism<3>::n_bands);  //--> new

    // The 2D Room_deism class
    py::class_<Room_deism<2>>(m, "Room2D_deism")
        //.def(py::init<py::list, py::list, const Eigen::MatrixXf &>())
        .def(py::init<const std::vector<Wall_deism<2>> &, const std::vector<int> &,
                        const std::vector<Microphone_deism<2>> &, float, int, float,
                        float, float, float, bool>())
        .def(py::init<const Vectorf<2> &,
                        const Eigen::Array<float, Eigen::Dynamic, 4> &,
                        const Eigen::Array<float, Eigen::Dynamic, 4> &,
                        const std::vector<Microphone_deism<2>> &, float, int, float,
                        float, float, float, bool>())
        .def(py::init<const std::vector<Wall_deism<2>> &, const std::vector<int> &,
                        const std::vector<Microphone_deism<2>> &, float, int, float,
                        float, float, float, bool, float>())   //tan new constructor←←
        .def(py::init<const Vectorf<2> &,
                        const Eigen::Array<float, Eigen::Dynamic, 4> &,
                        const Eigen::Array<float, Eigen::Dynamic, 4> &,
                        const std::vector<Microphone_deism<2>> &, float, int, float,
                        float, float, float, bool, float>())   //tan new constructor←←
        // .def("set_params", &Room_deism<2>::set_params)
        .def("set_params",static_cast<void (Room_deism<2>::*)(float,int,float,float,
                    float,float,bool)>(&Room_deism<2>::set_params))   //original
        .def("set_params",static_cast<void (Room_deism<2>::*)(float,int,float,float,
                    float,float,bool,float)>(&Room_deism<2>::set_params))     //tan overloaded
        .def("get_image_attenuation",&Room_deism<2>::get_image_attenuation)   //tan new added
        .def("add_mic", &Room_deism<2>::add_mic)
        .def("reset_mics", &Room_deism<2>::reset_mics)
        .def("image_source_model", &Room_deism<2>::image_source_model)
        .def("get_wall", &Room_deism<2>::get_wall)
        .def("get_max_distance", &Room_deism<2>::get_max_distance)
        .def("next_wall_hit", &Room_deism<2>::next_wall_hit)
        .def("scat_ray", &Room_deism<2>::scat_ray)
        .def("simul_ray", &Room_deism<2>::simul_ray)
        .def("ray_tracing",
            (void(Room_deism<2>::*)(
                const Eigen::Matrix<float, 1, Eigen::Dynamic> &angles,
                const Vectorf<2> source_pos)) &
                Room_deism<2>::ray_tracing)
        .def("ray_tracing", (void(Room_deism<2>::*)(size_t nb_phis, size_t nb_thetas,
                                                const Vectorf<2> source_pos)) &
                                Room_deism<2>::ray_tracing)
        .def("ray_tracing",
            (void(Room_deism<2>::*)(size_t n_rays, const Vectorf<2> source_pos)) &
                Room_deism<2>::ray_tracing)
        .def("contains", &Room_deism<2>::contains)
        .def("image_sources_dfs",(void(Room_deism<2>::*)(ImageSource<2> &is, int max_order))&Room_deism<2>::image_sources_dfs)      //-->new 
        .def("image_sources_dfs",(void(Room_deism<2>::*)(ImageSource<2> &is, int max_order,
            std::vector<Vectorf<2>>& list_intercep_p_to_is))&Room_deism<2>::image_sources_dfs)    //-->new
        // .def("is_visible_dfs",(bool (Room_deism<2>::*)(const Vectorf<2> &p, ImageSource<2> &is))&Room_deism<2>::is_visible_dfs)    //-->new
        .def("is_visible_dfs",(std::pair<bool,std::vector<Vectorf<2>>> (Room_deism<2>::*)(const Vectorf<2> &p, ImageSource<2> &is))&Room_deism<2>::is_visible_dfs)    //-->new
        .def("is_obstructed_dfs",&Room_deism<2>::is_obstructed_dfs)   //-->new
        .def("fill_sources",&Room_deism<2>::fill_sources)     //-->new
        .def_property_readonly_static("dim",
                                        [](py::object /* self */) { return 2; })
        .def_property("is_hybrid_sim", &Room_deism<2>::get_is_hybrid_sim,
                        &Room_deism<2>::set_is_hybrid_sim)
        .def_readonly("walls", &Room_deism<2>::walls)
        .def_readonly("sources", &Room_deism<2>::sources)
        .def_readonly("orders", &Room_deism<2>::orders)
        .def_readonly("orders_xyz", &Room_deism<2>::orders_xyz)
        .def_readonly("attenuations", &Room_deism<2>::attenuations)
        .def_readonly("gen_walls", &Room_deism<2>::gen_walls)
        .def_readonly("visible_mics", &Room_deism<2>::visible_mics)
        .def_readonly("walls", &Room_deism<2>::walls)
        .def_readonly("obstructing_walls", &Room_deism<2>::obstructing_walls)
        .def_readonly("microphones", &Room_deism<2>::microphones)
        .def_readonly("max_dist", &Room_deism<2>::max_dist)
        .def_readonly("impedance",&Room_deism<2>::impedance)  //--> new
        .def_readonly("reflection_matrix",&Room_deism<2>::reflection_matrix)  //--> new
        .def_readwrite("n_bands",&Room_deism<2>::n_bands);  //--> new

    // The Wall_deism class
    py::class_<Wall_deism<3>> wall_cls(m, "Wall_deism");

    wall_cls
        .def(py::init<const Eigen::Matrix<float, 3, Eigen::Dynamic> &,
                        const Eigen::ArrayXf &, const Eigen::ArrayXf &,
                        const std::string &>(),
            py::arg("corners"), py::arg("absorption") = Eigen::ArrayXf::Zero(1),
            py::arg("scattering") = Eigen::ArrayXf::Zero(1),
            py::arg("name") = "")
        .def(py::init<const Eigen::Matrix<float, 3, Eigen::Dynamic> &,
                        const Eigen::Matrix<float,3,1>&, 
                        float&,
                        const Eigen::ArrayXf &, const Eigen::ArrayXf &,
                        const std::string &>(),
            py::arg("corners"),py::arg("centroid"), py::arg("impedance"), 
            py::arg("absorption") = Eigen::ArrayXf::Zero(1),
            py::arg("scattering") = Eigen::ArrayXf::Zero(1),
            py::arg("name") = "")    //--> new
        .def("get_attenuation",&Wall_deism<3>::get_attenuation)   //--> new
        .def("area", &Wall_deism<3>::area)
        .def("intersection", &Wall_deism<3>::intersection)
        .def("intersects", &Wall_deism<3>::intersects)
        .def("side", &Wall_deism<3>::side)
        .def("reflect", &Wall_deism<3>::reflect)
        .def("normal_reflect",
            (Vectorf<3>(Wall_deism<3>::*)(const Vectorf<3> &, const Vectorf<3> &,
                                    float) const) &
                Wall_deism<3>::normal_reflect)
        .def("normal_reflect",
            (Vectorf<3>(Wall_deism<3>::*)(const Vectorf<3> &) const) &
                Wall_deism<3>::normal_reflect)
        .def("same_as", &Wall_deism<3>::same_as)
        .def_property_readonly_static("dim",
                                        [](py::object /* self */) { return 3; })
        .def_readwrite("absorption", &Wall_deism<3>::absorption)
        .def_readwrite("scatter", &Wall_deism<3>::scatter)
        .def_readwrite("name", &Wall_deism<3>::name)
        .def_readonly("corners", &Wall_deism<3>::corners)
        .def_readonly("origin", &Wall_deism<3>::origin)
        .def_readonly("normal", &Wall_deism<3>::normal)
        .def_readonly("basis", &Wall_deism<3>::basis)
        .def_readonly("flat_corners", &Wall_deism<3>::flat_corners)
        .def_readonly("reflection_matrix", &Wall_deism<3>::reflection_matrix)     //tan new
        .def_readonly("impedance", &Wall_deism<3>::impedance)     //tan new
        .def_readonly("centroid", &Wall_deism<3>::centroid);     //-->new

    py::enum_<Wall_deism<3>::Isect>(wall_cls, "Isect_deism")
        .value("NONE", Wall_deism<3>::Isect::NONE)
        .value("VALID", Wall_deism<3>::Isect::VALID)
        .value("ENDPT", Wall_deism<3>::Isect::ENDPT)
        .value("BNDRY", Wall_deism<3>::Isect::BNDRY)
        .export_values();

    // The Wall_deism class
    py::class_<Wall_deism<2>> wall2d_cls(m, "Wall2D_deism");

    wall2d_cls
        .def(py::init<const Eigen::Matrix<float, 2, Eigen::Dynamic> &,
                        const Eigen::ArrayXf &, 
                        const Eigen::ArrayXf &,
                        const std::string &>(),
            py::arg("corners"), py::arg("absorption") = Eigen::ArrayXf::Zero(1),
            py::arg("scattering") = Eigen::ArrayXf::Zero(1),
            py::arg("name") = "")
        .def(py::init<const Eigen::Matrix<float, 2, Eigen::Dynamic> &,
                        const Eigen::Matrix<float,2,1>&,
                        float&,
                        const Eigen::ArrayXf &, 
                        const Eigen::ArrayXf &,
                        const std::string & 
                        >(),
            py::arg("corners"),py::arg("centroid"), py::arg("impedance"), 
            py::arg("absorption") = Eigen::ArrayXf::Zero(1),
            py::arg("scattering") = Eigen::ArrayXf::Zero(1),
            py::arg("name") = "")    //--> new
        .def("get_attenuation",&Wall_deism<2>::get_attenuation)   //tan new
        .def("area", &Wall_deism<2>::area)
        .def("intersection", &Wall_deism<2>::intersection)
        .def("intersects", &Wall_deism<2>::intersects)
        .def("side", &Wall_deism<2>::side)
        .def("reflect", &Wall_deism<2>::reflect)
        .def("normal_reflect",
            (Vectorf<2>(Wall_deism<2>::*)(const Vectorf<2> &, const Vectorf<2> &,
                                    float) const) &
                Wall_deism<2>::normal_reflect)
        .def("normal_reflect",
            (Vectorf<2>(Wall_deism<2>::*)(const Vectorf<2> &) const) &
                Wall_deism<2>::normal_reflect)
        .def("same_as", &Wall_deism<2>::same_as)
        .def_property_readonly_static("dim",
                                        [](py::object /* self */) { return 2; })
        .def_readwrite("absorption", &Wall_deism<2>::absorption)
        .def_readwrite("scatter", &Wall_deism<2>::scatter)
        .def_readwrite("name", &Wall_deism<2>::name)
        .def_readonly("corners", &Wall_deism<2>::corners)
        .def_readonly("origin", &Wall_deism<2>::origin)
        .def_readonly("normal", &Wall_deism<2>::normal)
        .def_readonly("basis", &Wall_deism<2>::basis)
        .def_readonly("flat_corners", &Wall_deism<2>::flat_corners)
        .def_readonly("reflection_matrix", &Wall_deism<2>::reflection_matrix)     //--> new
        .def_readonly("impedance", &Wall_deism<2>::impedance)                    //--> new
        .def_readonly("centroid", &Wall_deism<2>::centroid);                    //--> new

    // The different wall intersection cases
    m.attr("WALL_ISECT_NONE") = WALL_ISECT_NONE;
    m.attr("WALL_ISECT_VALID") = WALL_ISECT_VALID;
    m.attr("WALL_ISECT_VALID_ENDPT") = WALL_ISECT_VALID_ENDPT;
    m.attr("WALL_ISECT_VALID_BNDRY") = WALL_ISECT_VALID_BNDRY;

    // The microphone class
    py::class_<Microphone_deism<3>>(m, "Microphone_deism")
        .def(py::init<const Vectorf<3> &, int, float, float>())
        .def_readonly("loc", &Microphone_deism<3>::loc)
        .def_readonly("hits", &Microphone_deism<3>::hits)
        .def_readonly("histograms", &Microphone_deism<3>::histograms);

    py::class_<Microphone_deism<2>>(m, "Microphone2D_deism")
        .def(py::init<const Vectorf<2> &, int, float, float>())
        .def_readonly("loc", &Microphone_deism<2>::loc)
        .def_readonly("hits", &Microphone_deism<2>::hits)
        .def_readonly("histograms", &Microphone_deism<2>::histograms);

    // The 2D histogram class
    py::class_<Histogram2D_deism>(m, "Histogram2D_deism")
        .def(py::init<int, int>())
        .def("log", &Histogram2D_deism::log)
        .def("bin", &Histogram2D_deism::bin)
        .def("get_hist", &Histogram2D_deism::get_hist)
        .def("reset", &Histogram2D_deism::reset);

    // Structure to hold detector hit information
    py::class_<Hit_deism>(m, "Hit_deism")
        .def(py::init<int>())
        .def(py::init<const float, const Eigen::ArrayXf &>())
        .def_readonly("transmitted", &Hit_deism::transmitted)
        .def_readonly("distance", &Hit_deism::distance);

    // getter and setter for geometric epsilon
    m.def("set_eps", [](const float &eps) { libroom_eps = eps; });
    m.def("get_eps", []() { return libroom_eps; });

    // Routines for the geometry packages
    m.def("ccw3p", &ccw3p, "Determines the orientation of three points");

    m.def("check_intersection_2d_segments", &check_intersection_2d_segments,
            "A function that checks if two line segments intersect");

    m.def("intersection_2d_segments", &intersection_2d_segments,
            "A function that finds the intersection of two line segments");

    m.def("intersection_3d_segment_plane", &intersection_3d_segment_plane,
            "A function that finds the intersection between a line segment and a "
            "plane");

    m.def("cross", &cross, "Cross product of two 3D vectors");

    m.def("is_inside_2d_polygon", &is_inside_2d_polygon,
            "Checks if a 2D point lies in or out of a planar polygon");

    m.def("area_2d_polygon", &area_2d_polygon,
            "Compute the signed area of a planar polygon");

    m.def("cos_angle_between", &cos_angle_between,
            "Computes the angle between two 2D or 3D vectors");

    m.def("dist_line_point", &dist_line_point,
            "Computes the distance between a point and an infinite line");

    m.def("rir_builder", &rir_builder, "RIR builder");
    m.def("delay_sum", &delay_sum, "Delay and sum");
    m.def("fractional_delay", &fractional_delay, "Fractional delays");
}
