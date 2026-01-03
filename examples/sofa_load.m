SOFAstart;
hrtf = sofaread('D:\Projects\DEISM-main\DEISM-main\examples\data\sampled_directivity\sofa\mit_kemar_normal_pinna.sofa');
%hrtf = SOFAload('D:\Projects\DEISM-main\DEISM-main\examples\data\sampled_directivity\sofa\mit_kemar_normal_pinna.sofa');
%SOFAinfo(hrtf);
%SOFAplotGeometry(hrtf);
%size(hrtf.Data.IR)
figure;
directivity(hrtf, 11111.1, Specification="measurements")