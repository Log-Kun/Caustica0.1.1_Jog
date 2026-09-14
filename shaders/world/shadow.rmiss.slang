// Shadow / sky-visibility miss (SBT miss index 1). Note this shader does not actually RUN in the
// current pipeline: visibility() uses a hit object purely as the traversal result and never invokes
// the miss shader — it fills the SBT slot. Kept behavior-identical to the GLSL original in case a
// future caller does invoke it (marks the ray escaped without touching the accumulated transmittance).
[shader("miss")]
void main(inout float4 shadowVis) {
    shadowVis.a = 1.0;
}
