// SPDX-License-Identifier: GPL-3.0-or-later
 // Copyright (C) 2026 Richard Myers

#pragma once

#include "thermomodel/model.hpp"
#include "thermomodel/derivatives.hpp"
#include "thermomodel/peng_robinson.hpp"

namespace thermogpu::thermo {

using State = thermomodel::State;
using Composition = thermomodel::Composition;
using Capabilities = thermomodel::Capabilities;
using Metadata = thermomodel::Metadata;

using Model = thermomodel::Model;
using DifferentiableModel = thermomodel::DifferentiableModel;
using Derivatives = thermomodel::Derivatives;

template<class T>
using Result = thermomodel::Result<T>;

using PengRobinson = thermomodel::PengRobinson;
using PRParameters = thermomodel::PRParameters;
using PRRootPolicy = thermomodel::PRRootPolicy;

} // namespace thermogpu::thermo
