"""Indian motor own-damage claim rules (IRDAI / India Motor Tariff conventions).

The figures below follow the commonly published IMT schedule. Verify them against
the current policy wording of the insurer before relying on the output.
"""

from dataclasses import dataclass

# Depreciation on parts by material, independent of vehicle age
MATERIAL_DEPRECIATION = {
    "rubber": 0.50,   # rubber / nylon / plastic parts, tyres, tubes, batteries
    "plastic": 0.50,
    "fibreglass": 0.30,
    "glass": 0.0,
}

# Depreciation on metal parts by vehicle age: (max age in years, rate)
METAL_DEPRECIATION = [
    (0.5, 0.0),
    (1, 0.05),
    (2, 0.10),
    (3, 0.15),
    (4, 0.25),
    (5, 0.35),
    (10, 0.40),
]
METAL_DEPRECIATION_OVER_10Y = 0.50

PAINT_MATERIAL_DEPRECIATION = 0.50
TOTAL_LOSS_THRESHOLD = 0.75  # repair cost above 75% of IDV => constructive total loss


def part_depreciation(material: str, vehicle_age_years: float) -> float:
    if material == "metal":
        for max_age, rate in METAL_DEPRECIATION:
            if vehicle_age_years <= max_age:
                return rate
        return METAL_DEPRECIATION_OVER_10Y
    return MATERIAL_DEPRECIATION[material]


def compulsory_deductible(engine_cc: int) -> int:
    return 1000 if engine_cc <= 1500 else 2000


@dataclass
class Policy:
    vehicle_age_years: float
    engine_cc: int = 1200
    idv: float | None = None          # Insured Declared Value; enables the total-loss check
    zero_dep: bool = False            # zero-depreciation (bumper-to-bumper) add-on
    voluntary_deductible: int = 0

    @property
    def deductible(self) -> int:
        return compulsory_deductible(self.engine_cc) + self.voluntary_deductible

    def depreciation(self, material: str) -> float:
        return 0.0 if self.zero_dep else part_depreciation(material, self.vehicle_age_years)

    @property
    def paint_depreciation(self) -> float:
        return 0.0 if self.zero_dep else PAINT_MATERIAL_DEPRECIATION
