#pragma once
#include <array>

template<size_t Dim>
class VirtualMap final {
	std::array<int, Dim> dimensions{};
	std::array<int, Dim> anchor{};

public:
	VirtualMap(std::array<int, Dim> dimensions) : dimensions(dimensions) {
	}

	void SetAnchor(std::array<int, Dim> anchor) {
		this->anchor = anchor;
	}

	std::array<int, Dim> GetPhysicalSlot(std::array<int, Dim> targetVirtualSlot) const {
		#define wrap(x, n) ((x % n) + n) % n
		std::array<int, Dim> physicalSlot;
		for (size_t i = 0; i < Dim; ++i) {
			int m = dimensions[i] / 2;
			int r = wrap(anchor[i] + targetVirtualSlot[i], dimensions[i]);
			physicalSlot[i] = r;
		}
		#undef wrap
		return physicalSlot;
	}
};