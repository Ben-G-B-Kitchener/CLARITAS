# V24.52 canonical finite acrylic topology

The acrylic body is the connected union of:

- sidewall: `Rin <= r <= Rout`, `Zmin <= z <= Zwall_top`
- bottom disc: `0 <= r <= Rout`, `Zlow <= z <= Zmin`

with `Zmean` (mean free-surface elevation) independent of `Zwall_top` (physical tube top).

`r=Rin` borders water below the local free surface and headspace air above it. The annular `z=Zmin`, `Rin<=r<=Rout` join is internal acrylic continuity: no Fresnel event and no RNG draw. The annular upper face at `Zwall_top` is acrylic/external-air. The bore `r<Rin` is open at `Zwall_top` and headspace air passes to external air without Fresnel processing.
