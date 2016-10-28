#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


/**
 *
 * def insert_trans_rot_frame(vf, f_pad, dx, dy, th):
    sin_th = np.sin(th)
    cos_th = np.cos(th)

    cx = int(np.floor(0.5 * f_pad.shape[0]) + dx)
    cy = int(np.floor(0.5 * f_pad.shape[1]) + dy)

    xr = cos_th * vf[:, 0] - sin_th * vf[:, 1]
    yr = sin_th * vf[:, 0] + cos_th * vf[:, 1]

    for i in range(vf.shape[0]):
        ind_x = int(np.floor(xr[i]))
        ind_y = int(np.floor(yr[i]))

        frac_x = xr[i] - ind_x
        frac_y = yr[i] - ind_y

        # f_pad[int (cx + ind_x), int (cy - ind_y)] += vf[i, 2]

        corn1_x = (cx + ind_x) % f_pad.shape[0]
        corn1_y = (cy - ind_y) % f_pad.shape[1]

        corn2_x = (cx + ind_x + 1) % f_pad.shape[0]
        corn2_y = (cy - ind_y - 1) % f_pad.shape[1]

        # bilinear interpolation
        f_pad[corn1_x, corn1_y] += (1.0 - frac_x) * (1.0 - frac_y) * vf[i, 2]
        f_pad[corn2_x, corn1_y] += frac_x * (1.0 - frac_y) * vf[i, 2]
        f_pad[corn1_x, corn2_y] += (1.0 - frac_x) * frac_y * vf[i, 2]
        f_pad[corn2_x, corn2_y] += frac_x * frac_y * vf[i, 2]
 *
 *
 */

void insert_trans_rot_frame(py::array_t<double> &vf, py::array_t<double> &f_pad, double dx, double dy, double th) {
    const auto sin_th = sin(th);
    const auto cos_th = cos(th);

    const auto info_vf = vf.request();
    const auto info_fpad = f_pad.request();

    const auto cx = (floor(0.5 * info_fpad.shape[0]) + dx);
    const auto cy = (floor(0.5 * info_fpad.shape[1]) + dy);

    std::vector<double> xr, yr;
    xr.reserve(info_vf.shape[0]);
    yr.reserve(info_vf.shape[0]);
    double** data_vf = static_cast<double**>(info_vf.ptr);
    double** data_fpad = static_cast<double**>(info_vf.ptr);
    for(std::size_t i = 0; i < info_vf.shape[0]; ++i) {
        xr.push_back(cos_th * data_vf[i][0] - sin_th * data_vf[i][1]);
        yr.push_back(sin_th * data_vf[i][0] + cos_th * data_vf[i][1]);
    }

    for (std::size_t i = 0; i < info_vf.shape[0]; ++i) {
        const auto ind_x = floor(xr[i]);
        const auto ind_y = floor(yr[i]);

        const auto frac_x = xr[i] - ind_x;
        const auto frac_y = yr[i] - ind_y;

        const auto corn1_x = static_cast<int>(cx + ind_x) % info_fpad.shape[0];
        const auto corn1_y = static_cast<int>(cy - ind_y) % info_fpad.shape[1];

        const auto corn2_x = static_cast<int>(cx + ind_x + 1) % info_fpad.shape[0];
        const auto corn2_y = static_cast<int>(cy - ind_y - 1) % info_fpad.shape[1];

        // bilinear interpolation
        data_fpad[corn1_x][corn1_y] += (1.0 - frac_x) * (1.0 - frac_y) * data_vf[i][2];
        data_fpad[corn2_x][corn1_y] += frac_x * (1.0 - frac_y) * data_vf[i][2];
        data_fpad[corn1_x][corn2_y] += (1.0 - frac_x) * frac_y * data_vf[i][2];
        data_fpad[corn2_x][corn2_y] += frac_x * frac_y * data_vf[i][2];
    }
}


PYBIND11_PLUGIN(super_resolution_tools) {
    py::module m("super_resolution_tools");

    m.def("insert_trans_rot_frame", &insert_trans_rot_frame);

    return m.ptr();
}