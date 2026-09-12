# V24.52 Roughness / optical-physics provenance

V24.52 changes apparatus geometry/topology, not the established particle or rough dielectric optical law. The retained Smith-Beckmann dielectric microsurface implementation, roughness parameters, Mie/hybrid event-rate physics and particle optical constants are not tuned to the new geometry or to sediment outputs.

No H170/material/PSD/concentration multiplier, empirical scatter boost or fitted detector gain is introduced. Geometry corrections may legitimately change which real physical interface a ray reaches and therefore may alter downstream RNG history, but an internal acrylic/acrylic continuation still consumes zero Fresnel RNG draws and real interfaces retain their defined draw ordering.
