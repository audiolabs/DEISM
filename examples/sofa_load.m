SOFAstart;
hrtf = SOFAload('D:\Projects\DEISM-main\DEISM-main\examples\data\sampled_directivity\sofa\BuK-ED_hrir.sofa');
%SOFAinfo(hrtf);
%SOFAplotGeometry(hrtf);
%size(hrtf.Data.IR)
figure;
directivity(s, 400, Specification="measurements")