
#pragma once

#include <tuple>

enum class SigmoidType {
  Gaussian,
  Hyperbolic,
  LongTail,
  Reciprocal,
  Cosine,
  Linear,
  Quadratic,
  TanhSquared
};

double hamacher_product(double a, double b);
double rect_prism_tolerance(const double *curr, const double *zero,
                            const double *one);
double tolerance(double x, std::tuple<double, double> bounds, double margin,
                 SigmoidType sigmoid);
double _sigmoids(double x, double value_at_1, SigmoidType sigmoid);
