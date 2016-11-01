#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

void remove_sprinkles(py::array_t<double> &target, py::array_t<double>& frames) {
    const auto info_vf = frames.request(false);
    const auto w = info_vf.shape[0];

    auto data_input = (double(*)[w]) frames.data(0);
    auto data_target = (double(*)[w]) target.data(0);

    for(int x = 1; x < w-1; ++x) {
        for(int y = 1; y < w-1; ++y) {

            int n_neighbors = 0;
            for(int dx = -1; dx < 1; ++dx) {
                for(int dy = -1; dy < 1; ++dy) {
                    if(!(dx == 0 && dy == 0)) {
                        if(data_input[x + dx][y + dy] != 0) {
                            ++n_neighbors;
                        }
                    }
                }
            }
            if(n_neighbors >= 3) {
                data_target[x][y] = data_input[x][y];
            } else {
                data_target[x][y] = 0;
            }

        }
    }
}

void median_noise_reduction(py::array_t<double> &target, py::array_t<double>& frames, const std::size_t window_width, const std::size_t window_height) {

    const auto info_target = target.request(true);
    const auto info_vf = frames.request(false);

    std::vector<double> window;
    window.resize(window_width * window_height);

    const auto edge_x = floor(window_width / 2.);
    const auto edge_y = floor(window_height / 2.);

    auto data_input = (double(*)[372]) frames.data(0);
    auto data_target = (double(*)[372]) target.data(0);

    for(int x = 1; x < 371; ++x) {
        for(int y = 1; y < 371; ++y) {
            int n_neighbors = 0;
            for(int dx = -1; dx < 1; ++dx) {
                for(int dy = -1; dy < 1; ++dy) {
                    if(!(dx == 0 && dy == 0)) {
                        if(data_input[x + dx][y + dy] != 0) {
                            ++n_neighbors;
                        }
                    }
                }
            }
            if(n_neighbors <= 4) {

            }
        }
    }

    for(auto x = edge_x; x < info_vf.shape[1]; ++x) {
        for(auto y = edge_y; y < info_vf.shape[0]; ++y) {
            std::size_t i = 0;
            for(auto fx = 0; fx < window_width; ++ fx) {
                for(auto fy = 0; fy < window_height; ++fy) {
                    window[i] = data_input[static_cast<std::size_t>(x + fx - edge_x)][static_cast<std::size_t>(y + fy - edge_y)];
                    ++i;
                }
            }
            // sort entries in window[]
            std::sort(window.begin(), window.end());
            data_target[(int) x][(int) y] = window[(int) (.5 * window_width * window_height)];
            // outputPixelValue[x][y] := window[window width * window height / 2]
        }
    }
}

void insert_trans_rot_frame(py::array_t<double> &vf, py::array_t<double> &f_pad, double dx, double dy, double th, double scale) {
    const auto sin_th = sin(th);
    const auto cos_th = cos(th);

    const auto info_vf = vf.request(true);
    const auto info_fpad = f_pad.request(true);

    const auto cx = (floor(0.5 * info_fpad.shape[0]) + dx);
    const auto cy = (floor(0.5 * info_fpad.shape[1]) + dy);

    std::vector<double> xr, yr;
    xr.reserve(info_vf.shape[0]);
    yr.reserve(info_vf.shape[0]);
    double *data_vf = vf.mutable_data(0);
    double *data_fpad = f_pad.mutable_data(0);
    for (std::size_t i = 0; i < info_vf.shape[0]; ++i) {
        xr.push_back(scale*(cos_th * data_vf[info_vf.shape[1] * i + 0] - sin_th * data_vf[info_vf.shape[1] * i + 1]));
        yr.push_back(scale*(sin_th * data_vf[info_vf.shape[1] * i + 0] + cos_th * data_vf[info_vf.shape[1] * i + 1]));
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
        data_fpad[corn1_x * info_fpad.shape[1] + corn1_y] += (1.0 - frac_x) * (1.0 - frac_y) * data_vf[info_vf.shape[1] * i + 2];
        data_fpad[corn2_x * info_fpad.shape[1] + corn1_y] += frac_x * (1.0 - frac_y) * data_vf[info_vf.shape[1] * i + 2];
        data_fpad[corn1_x * info_fpad.shape[1] + corn2_y] += (1.0 - frac_x) * frac_y * data_vf[info_vf.shape[1] * i + 2];
        data_fpad[corn2_x * info_fpad.shape[1] + corn2_y] += frac_x * frac_y * data_vf[info_vf.shape[1] * i + 2];
    }
}

double get_idx(py::array_t<double> &arr, std::size_t ix, std::size_t iy) {
    auto r = arr.request();
    return arr.data(0)[r.shape[1]*ix + iy];
}


PYBIND11_PLUGIN(super_resolution_tools) {
    py::module m("super_resolution_tools");

    m.def("insert_trans_rot_frame", &insert_trans_rot_frame);
    m.def("median_noise_reduction", &median_noise_reduction);
    m.def("remove_sprinkles", &remove_sprinkles);
    m.def("get_idx", &get_idx);

    return m.ptr();
}