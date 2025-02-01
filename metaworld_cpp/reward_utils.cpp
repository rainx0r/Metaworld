#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

#define DEFAULT_VALUE_AT_MARGIN 0.1

namespace {
double in_range(double a, double b, double c) {
  return (c >= b) ? (double)(b <= a && a <= c) : (double)(c <= a && a <= b);
}
} // namespace

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

double hamacher_product(double a, double b) {
  if (!(0.0 <= a && a <= 1.0 && 0.0 <= b && b <= 1.0)) {
    throw std::invalid_argument("a (" + std::to_string(a) + ") and b (" +
                                std::to_string(b) +
                                ") must range between 0 and 1");
  }

  double denominator = a + b - (a * b);
  double h_prod = (denominator > 0) ? ((a * b) / denominator) : 0.0;

  if (!(0.0 <= h_prod && h_prod <= 1.0)) {
    throw std::runtime_error(
        "hamacher product result out of valid range [0,1]");
  }
  return h_prod;
}

// TODO: the double* here might wanna be vec3 eventually
double rect_prism_tolerance(const double *curr, const double *zero,
                            const double *one) {
  bool in_prism = in_range(curr[0], zero[0], one[0]) &&
                  in_range(curr[1], zero[1], one[1]) &&
                  in_range(curr[2], zero[2], one[2]);

  if (in_prism) {
    double diff[3] = {one[0] - zero[0], one[1] - zero[1], one[2] - zero[2]};
    double x_scale = (curr[0] - zero[0]) / diff[0];
    double y_scale = (curr[1] - zero[1]) / diff[1];
    double z_scale = (curr[2] - zero[2]) / diff[2];
    return x_scale * y_scale * z_scale;
  }
  return 1.0;
}

double _sigmoids(double x, double value_at_1, SigmoidType sigmoid) {
  // Validate value_at_1 based on sigmoid type
  if (sigmoid == SigmoidType::Cosine || sigmoid == SigmoidType::Linear ||
      sigmoid == SigmoidType::Quadratic) {
    if (!(0.0 <= value_at_1 && value_at_1 < 1.0)) {
      throw std::invalid_argument(
          "value_at_1 must be nonnegative and smaller than 1, got " +
          std::to_string(value_at_1));
    }
  } else {
    if (!(0.0 < value_at_1 && value_at_1 < 1.0)) {
      throw std::invalid_argument(
          "value_at_1 must be strictly between 0 and 1, got " +
          std::to_string(value_at_1));
    }
  }

  double scale;
  double scaled_x;

  switch (sigmoid) {
  case SigmoidType::Gaussian:
    scale = std::sqrt(-2.0 * std::log(value_at_1));
    return std::exp(-0.5 * std::pow(x * scale, 2));

  case SigmoidType::Hyperbolic:
    scale = std::acosh(1.0 / value_at_1);
    return 1.0 / std::cosh(x * scale);

  case SigmoidType::LongTail:
    scale = std::sqrt(1.0 / value_at_1 - 1.0);
    return 1.0 / (std::pow(x * scale, 2) + 1.0);

  case SigmoidType::Reciprocal:
    scale = 1.0 / value_at_1 - 1.0;
    return 1.0 / (std::abs(x) * scale + 1.0);

  case SigmoidType::Cosine:
    scale = std::acos(2.0 * value_at_1 - 1.0) / M_PI;
    scaled_x = x * scale;
    return std::abs(scaled_x) < 1.0 ? (1.0 + std::cos(M_PI * scaled_x)) / 2.0
                                    : 0.0;

  case SigmoidType::Linear:
    scale = 1.0 - value_at_1;
    scaled_x = x * scale;
    return std::abs(scaled_x) < 1.0 ? 1.0 - scaled_x : 0.0;

  case SigmoidType::Quadratic:
    scale = std::sqrt(1.0 - value_at_1);
    scaled_x = x * scale;
    return std::abs(scaled_x) < 1.0 ? 1.0 - std::pow(scaled_x, 2) : 0.0;

  case SigmoidType::TanhSquared:
    scale = std::atanh(std::sqrt(1.0 - value_at_1));
    return 1.0 - std::pow(std::tanh(x * scale), 2);

  default:
    throw std::invalid_argument("Unknown sigmoid type");
  }
}

double tolerance(double x, std::tuple<double, double> bounds, double margin,
                 SigmoidType sigmoid) {
  double lower = std::get<0>(bounds);
  double upper = std::get<1>(bounds);

  // Validate bounds
  if (lower > upper) {
    throw std::invalid_argument("Lower bound must be <= upper bound");
  }
  if (margin < 0) {
    throw std::invalid_argument("margin must be non-negative. Current value: " +
                                std::to_string(margin));
  }

  // Check if x is in bounds
  bool in_bounds = (lower <= x) && (x <= upper);

  // If margin is 0, return binary result
  if (margin == 0.0) {
    return in_bounds ? 1.0 : 0.0;
  }

  // Calculate distance to nearest bound
  double d;
  if (x < lower) {
    d = (lower - x) / margin;
  } else {
    d = (x - upper) / margin;
  }

  // Return result
  return in_bounds ? 1.0 : _sigmoids(d, DEFAULT_VALUE_AT_MARGIN, sigmoid);
}
