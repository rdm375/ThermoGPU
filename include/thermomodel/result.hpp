#pragma once
#include <string>
#include <utility>

namespace thermomodel {
enum class ErrorCode { None, InvalidInput, OutOfRange, Unsupported, NoConvergence, AmbiguousState };
template<class T> struct Result {
    T value{};
    ErrorCode error = ErrorCode::None;
    std::string message;
    explicit operator bool() const noexcept { return error == ErrorCode::None; }
    static Result success(T v) { Result r; r.value = std::move(v); return r; }
    static Result failure(ErrorCode e, std::string msg) { Result r; r.error=e; r.message=std::move(msg); return r; }
};
} // namespace thermomodel
