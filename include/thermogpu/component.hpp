// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include <string_view>
namespace thermogpu { struct Component { std::string_view name; double critical_temperature_K; double critical_pressure_Pa; double acentric_factor; double molar_mass_kg_per_mol; }; }
