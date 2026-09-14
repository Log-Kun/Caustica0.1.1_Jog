
import world_common;

[[vk::push_constant]] WorldPushConstants pc;


static const float PI = 3.14159265359;
float pow2(float x) { return x * x; }

static const float3 SUN_DISC_RADIANCE  = float3(24.0, 23.0, 21.5);
static const float3 MOON_DISC_RADIANCE = float3(1.7, 1.85, 2.2);

static const float SKY_SATURATION = 1.2;

static const float SUN_DISC_HALF_ANGLE  = 0.036;
static const float MOON_DISC_HALF_ANGLE = 0.032;

static const float3 NIGHT_ZENITH  = float3(0.004, 0.008, 0.022) / 4.0;
static const float3 NIGHT_HORIZON = float3(0.015, 0.022, 0.045) / 4.0;

static const float  PLANET_R   = 6371000.0;
static const float  ATMOS_R    = 6471000.0;
static const float3 RAY_BETA   = float3(5.5e-6, 13.0e-6, 22.4e-6);
static const float  MIE_BETA   = 21.0e-6;
static const float  RAY_SCALE_H = 8000.0;
static const float  MIE_SCALE_H = 1200.0;
static const float  MIE_G      = 0.758;

static const float3 OZONE_BETA = float3(0.650e-6, 1.881e-6, 0.085e-6);
static const float  OZONE_CENTER_H = 25000.0;
static const float  OZONE_WIDTH_H  = 15000.0;
static const float  SUN_INTENSITY = 22.0;
static const int    ATMOS_PRIMARY_STEPS = 16;
static const int    ATMOS_LIGHT_STEPS   = 8;

float ozoneDensity(float h) { return max(0.0, 1.0 - abs(h - OZONE_CENTER_H) / OZONE_WIDTH_H); }

float2 raySphere(float3 o, float3 d, float r) {
    float b = dot(o, d);
    float c = dot(o, o) - r * r;
    float disc = b * b - c;
    if (disc < 0.0) return float2(1.0e9, -1.0e9);
    disc = sqrt(disc);
    return float2(-b - disc, -b + disc);
}

float3 atmosphere(float3 dir, float3 sunDir) {
    float3 orig = float3(0.0, PLANET_R + 2000.0, 0.0);
    float2 t = raySphere(orig, dir, ATMOS_R);
    if (t.x > t.y) return float3(0.0, 0.0, 0.0);
    float tStart = max(t.x, 0.0);
    float tEnd = t.y;
    float2 tGround = raySphere(orig, dir, PLANET_R);
    if (tGround.x > 0.0) tEnd = min(tEnd, tGround.x);
    float segLen = (tEnd - tStart) / float(ATMOS_PRIMARY_STEPS);

    float mu = dot(dir, sunDir);
    float phaseR = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float g = MIE_G;
    float phaseM = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + mu * mu))
                 / ((2.0 + g * g) * pow(max(1.0 + g * g - 2.0 * g * mu, 0.0), 1.5));

    float3 sumR = float3(0.0, 0.0, 0.0), sumM = float3(0.0, 0.0, 0.0);
    float odR = 0.0, odM = 0.0, odO = 0.0;
    float tCur = tStart;
    for (int i = 0; i < ATMOS_PRIMARY_STEPS; i++) {
        float3 p = orig + dir * (tCur + segLen * 0.5);
        float h = length(p) - PLANET_R;
        float hr = exp(-h / RAY_SCALE_H) * segLen;
        float hm = exp(-h / MIE_SCALE_H) * segLen;
        odR += hr; odM += hm; odO += ozoneDensity(h) * segLen;

        float2 tl = raySphere(p, sunDir, ATMOS_R);
        float segLenL = tl.y / float(ATMOS_LIGHT_STEPS);
        float tlCur = 0.0, odRL = 0.0, odML = 0.0, odOL = 0.0;
        bool blocked = false;
        for (int j = 0; j < ATMOS_LIGHT_STEPS; j++) {
            float3 pl = p + sunDir * (tlCur + segLenL * 0.5);
            float hl = length(pl) - PLANET_R;
            if (hl < 0.0) { blocked = true; break; }
            odRL += exp(-hl / RAY_SCALE_H) * segLenL;
            odML += exp(-hl / MIE_SCALE_H) * segLenL;
            odOL += ozoneDensity(hl) * segLenL;
            tlCur += segLenL;
        }
        if (!blocked) {
            float3 tau = RAY_BETA * (odR + odRL) + MIE_BETA * 1.1 * (odM + odML)
                       + OZONE_BETA * (odO + odOL);
            float3 atten = exp(-tau);
            sumR += hr * atten;
            sumM += hm * atten;
        }
        tCur += segLen;
    }
    return SUN_INTENSITY * (sumR * RAY_BETA * phaseR + sumM * MIE_BETA * phaseM);
}

float3 transmittanceToSpace(float3 dir) {
    float3 orig = float3(0.0, PLANET_R + 2000.0, 0.0);
    float2 t = raySphere(orig, dir, ATMOS_R);
    if (t.y <= 0.0) return float3(1.0, 1.0, 1.0);
    float segLen = t.y / float(ATMOS_LIGHT_STEPS);
    float odR = 0.0, odM = 0.0, odO = 0.0;
    for (int i = 0; i < ATMOS_LIGHT_STEPS; i++) {
        float3 p = orig + dir * (segLen * (float(i) + 0.5));
        float h = length(p) - PLANET_R;
        odR += exp(-h / RAY_SCALE_H) * segLen;
        odM += exp(-h / MIE_SCALE_H) * segLen;
        odO += ozoneDensity(h) * segLen;
    }
    return exp(-(RAY_BETA * odR + MIE_BETA * 1.1 * odM + OZONE_BETA * odO));
}

void addCelestialDisc(inout float3 col, float3 dir, float3 bodyDir, float halfAngle, float3 radiance, float weight) {
    float bodyUp = clamp((bodyDir.y + 0.0625) / 0.125, 0.0, 1.0);
    float bodyDist = acos(clamp(dot(dir, bodyDir), -1.0, 1.0));
    float bodyMask = 1.0 - smoothstep(halfAngle * 0.82, halfAngle, bodyDist);
    if (bodyMask > 0.0 && bodyUp > 0.0) {
        col += radiance * transmittanceToSpace(dir) * bodyMask * bodyUp * weight;
    }
}

float hash13(float3 p3) {
    p3 = frac(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

float3 rotateAxis(float3 v, float3 axis, float ang) {
    float c = cos(ang), s = sin(ang);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

static const float STAR_CELLS     = 21.0;
static const float STAR_THRESHOLD = 0.86;
static const float STAR_HALF      = 0.045;
float3 stars(float3 dir, float starBrightness, float VdotS, float4 celestial) {
    if (starBrightness <= 0.0 || dir.y <= 0.0) return float3(0.0, 0.0, 0.0);

    float3 sdir = rotateAxis(dir, normalize(celestial.xyz), -celestial.w);

    float3 a = abs(sdir);
    float2 uv;
    float face;
    if (a.x >= a.y && a.x >= a.z)      { uv = sdir.yz / a.x; face = sdir.x > 0.0 ? 0.0 : 1.0; }
    else if (a.y >= a.z)               { uv = sdir.xz / a.y; face = sdir.y > 0.0 ? 2.0 : 3.0; }
    else                               { uv = sdir.xy / a.z; face = sdir.z > 0.0 ? 4.0 : 5.0; }
    float2 g = uv * STAR_CELLS;
    float3 id = float3(floor(g), face);
    if (hash13(id) < STAR_THRESHOLD) return float3(0.0, 0.0, 0.0);

    float sz = STAR_HALF * (0.6 + 0.8 * hash13(id + 53.0));
    float inset = sz * 1.5;
    float2 center = inset + (1.0 - 2.0 * inset) * float2(hash13(id + 11.0), hash13(id + 23.0));
    float2 q = frac(g) - center;
    float ang = hash13(id + 91.0) * 6.2831853;
    float cs = cos(ang), sn = sin(ang);
    q = float2(cs * q.x - sn * q.y, sn * q.x + cs * q.y);
    float box = max(abs(q.x), abs(q.y));
    float star = 1.0 - smoothstep(sz * 0.6, sz, box);
    if (star <= 0.0) return float3(0.0, 0.0, 0.0);
    float bright = 0.55 + 0.45 * hash13(id + 71.0);
    star *= bright * min(dir.y * 3.0, 1.0) * max(0.0, 1.0 - pow(abs(VdotS) * 1.002, 100.0));

    float3 starCol = lerp(float3(0.78, 0.60, 0.42), float3(0.50, 0.62, 0.95), hash13(id + 137.0));
    return 0.2 * star * starCol * starBrightness;
}

[shader("miss")]
void main(inout Payload payload) {
    WorldPush worldPush = ConstPtr<WorldPush>(pc.worldPushAddr)[0];
    float3 dir = normalize(WorldRayDirection());
    float day = worldPush.sunDir.w;
    float3 sd = worldPush.sunDir.xyz;
    float SdotU = sd.y;
    float VdotS = dot(dir, sd);
    float sunUp = clamp((SdotU + 0.0625) / 0.125, 0.0, 1.0);
    float starBrightness = worldPush.lightRadiance.w;
    bool showCelestial = (payload.flags & PAYLOAD_SHOW_CELESTIAL) != 0u;

    float3 mdir = dir;
    float downFade = 1.0;
    if (dir.y < 0.0) {
        mdir = normalize(float3(dir.x, 0.0, dir.z));
        downFade = exp(dir.y * 3.0);
    }
    float3 col = atmosphere(mdir, sd);

    float lum = dot(col, float3(0.2126, 0.7152, 0.0722));
    col = max(lerp(float3(lum, lum, lum), col, SKY_SATURATION), float3(0.0, 0.0, 0.0));
    float up = clamp(dir.y, 0.0, 1.0);
    float3 nightAmbient = lerp(NIGHT_HORIZON, NIGHT_ZENITH, up);
    col += nightAmbient * (1.0 - day);
    col *= downFade;

    col += stars(dir, starBrightness, VdotS, worldPush.celestial);

    if (showCelestial) {
        addCelestialDisc(col, dir, sd, SUN_DISC_HALF_ANGLE, SUN_DISC_RADIANCE, 1.0);
        addCelestialDisc(col, dir, normalize(worldPush.moonDir.xyz), MOON_DISC_HALF_ANGLE, MOON_DISC_RADIANCE, pow2(1.0 - day));
    }

    payload.albedo = max(col, float3(0.0, 0.0, 0.0));
    payload.hitT = -1.0;
    payload.normal = float3(0.0, 0.0, 0.0);
    payload.motionPrev = float3(0.0, 0.0, 0.0);
    payload.f0 = float3(0.0, 0.0, 0.0);
    payload.flags = 0u;
    payload.roughMetal = packHalf2(float2(1.0, 0.0));
    payload.emissionSss = packHalf2(float2(0.0, 0.0));
    payload.iorTransmission = packHalf2(float2(1.0, 0.0));
    payload.rayCone = 0u;
}
