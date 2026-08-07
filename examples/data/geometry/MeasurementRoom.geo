Point(1) = { 0.000000, 5.100000, 0.000000, 1.0 };
Point(2) = { 6.210000, 4.000000, 0.000000, 1.0 };
Point(3) = { 5.520000, 0.000000, 0.000000, 1.0 };
Point(4) = { 0.000000, 0.000000, 0.000000, 1.0 };
Point(5) = { 0.000000, 5.100000, 3.300000, 1.0 };
Point(6) = { 6.210000, 4.000000, 3.300000, 1.0 };
Point(7) = { 0.000000, 0.000000, 3.300000, 1.0 };
Point(8) = { 5.520000, 0.000000, 3.300000, 1.0 };

Line(1) = { 1, 2 };
Line(2) = { 1, 4 };
Line(3) = { 1, 5 };
Line(4) = { 2, 3 };
Line(5) = { 2, 6 };
Line(6) = { 3, 4 };
Line(7) = { 3, 8 };
Line(8) = { 4, 7 };
Line(9) = { 5, 6 };
Line(10) = { 5, 7 };
Line(11) = { 6, 8 };
Line(12) = { 7, 8 };

Line Loop(1) = { 6, -2, 1, 4 };
Line Loop(2) = { -1, 3, 9, -5 };
Line Loop(3) = { -9, 10, 12, -11 };
Line Loop(4) = { -6, 7, -12, -8 };
Line Loop(5) = { 2, 8, -10, -3 };
Line Loop(6) = { 11, -7, -4, 5 };

Plane Surface(1) = { 1 };
Plane Surface(2) = { 2 };
Plane Surface(3) = { 3 };
Plane Surface(4) = { 4 };
Plane Surface(5) = { 5 };
Plane Surface(6) = { 6 };

Surface Loop(1) = { 1, 2, 3, 4, 5, 6 };
Physical Surface("floor") = { 1 };
Physical Surface("wall1") = { 2 };
Physical Surface("ceiling") = { 3 };
Physical Surface("wall2") = { 4 };
Physical Surface("wall3") = { 5 };
Physical Surface("wall4") = { 6 };
Volume( 1 ) = { 1 };
Physical Volume("RoomVolume") = { 1 };
Physical Line ("default") = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1; // Delaunay3D, works for boundary layer insertion.
Mesh.Optimize = 1; // Gmsh smoother, works with boundary layers (netgen version does not).
Mesh.CharacteristicLengthFromPoints = 1;
// Recombine Surface "*";
