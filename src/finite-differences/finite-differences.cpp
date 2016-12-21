#include <pybind11/common.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void solve(std::function<double(double)> &b, std::function<double(double)> &c, std::function<double(double)> &f, unsigned int N, double a, double b) {

}


PYBIND11_PLUGIN(finite_differences) {
        py::module m("finite_differences");

        return m.ptr();
}