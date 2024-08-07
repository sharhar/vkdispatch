#version 450

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer DataIn{
	vec2 inputs[65536];
};

layout(std430, binding = 1) buffer DataOut{
	vec2 outputs[65536];
};

shared vec2 sdata[272];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts

void main() {
    vec2 temp_0 = vec2(0.0f);
    vec2 temp_1 = vec2(0.0f);
    vec2 temp_2 = vec2(0.0f);
    vec2 temp_3 = vec2(0.0f);
    vec2 temp_4 = vec2(0.0f);
    vec2 temp_5 = vec2(0.0f);
    vec2 temp_6 = vec2(0.0f);
    vec2 temp_7 = vec2(0.0f);
    vec2 w = vec2(0.0f);
    vec2 loc_0 = vec2(0.0f);
    uint tempInt = 0;
    uint tempInt2 = 0;
    uint shiftX = gl_WorkGroupID.x;
    uint shiftY = gl_WorkGroupID.y;
    uint shiftZ = 0;
    vec2 iw = 0;
    uint stageInvocationID = 0;
    uint blockInvocationID = 0;
    uint sdataID = 0;
    uint combinedID = 0;
    uint inoutID = 0;
    uint inoutID_x = 0;
    uint inoutID_y = 0;
    float angle = 0;

    inoutID = 256 * shiftY;
    inoutID = inoutID + shiftZ;
    inoutID = inoutID + gl_LocalInvocationID.x;
    combinedID = gl_LocalInvocationID.x + 0;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    temp_0 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 32;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_1 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 64;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_2 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 96;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_3 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 128;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_4 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 160;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_5 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 192;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_6 = inputs[inoutID];
    combinedID = gl_LocalInvocationID.x + 224;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    temp_7 = inputs[inoutID];
    stageInvocationID = gl_LocalInvocationID.x + 0;
    stageInvocationID = stageInvocationID % 1;
    angle = stageInvocationID * -3.14159265358979312e+00f;
    w.x = 1.00000000000000000e+00f;
    w.y = 0.00000000000000000e+00f;
    loc_0.x = temp_4.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_4.x, w.x, loc_0.x);
    loc_0.y = temp_4.y * w.x;
    loc_0.y = fma(temp_4.x, w.y, loc_0.y);
    temp_4.x = temp_0.x - loc_0.x;
    temp_4.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = temp_5.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_5.x, w.x, loc_0.x);
    loc_0.y = temp_5.y * w.x;
    loc_0.y = fma(temp_5.x, w.y, loc_0.y);
    temp_5.x = temp_1.x - loc_0.x;
    temp_5.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    loc_0.x = temp_6.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, w.x, loc_0.x);
    loc_0.y = temp_6.y * w.x;
    loc_0.y = fma(temp_6.x, w.y, loc_0.y);
    temp_6.x = temp_2.x - loc_0.x;
    temp_6.y = temp_2.y - loc_0.y;
    temp_2.x = temp_2.x + loc_0.x;
    temp_2.y = temp_2.y + loc_0.y;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_3.x - loc_0.x;
    temp_7.y = temp_3.y - loc_0.y;
    temp_3.x = temp_3.x + loc_0.x;
    temp_3.y = temp_3.y + loc_0.y;
    w.x = 1.00000000000000000e+00f;
    w.y = 0.00000000000000000e+00f;
    loc_0.x = temp_2.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_2.x, w.x, loc_0.x);
    loc_0.y = temp_2.y * w.x;
    loc_0.y = fma(temp_2.x, w.y, loc_0.y);
    temp_2.x = temp_0.x - loc_0.x;
    temp_2.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = temp_3.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_3.x, w.x, loc_0.x);
    loc_0.y = temp_3.y * w.x;
    loc_0.y = fma(temp_3.x, w.y, loc_0.y);
    temp_3.x = temp_1.x - loc_0.x;
    temp_3.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    iw.x = w.y;
    iw.y = -w.x;
    loc_0.x = temp_6.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, iw.x, loc_0.x);
    loc_0.y = temp_6.y * iw.x;
    loc_0.y = fma(temp_6.x, iw.y, loc_0.y);
    temp_6.x = temp_4.x - loc_0.x;
    temp_6.y = temp_4.y - loc_0.y;
    temp_4.x = temp_4.x + loc_0.x;
    temp_4.y = temp_4.y + loc_0.y;
    loc_0.x = temp_7.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, iw.x, loc_0.x);
    loc_0.y = temp_7.y * iw.x;
    loc_0.y = fma(temp_7.x, iw.y, loc_0.y);
    temp_7.x = temp_5.x - loc_0.x;
    temp_7.y = temp_5.y - loc_0.y;
    temp_5.x = temp_5.x + loc_0.x;
    temp_5.y = temp_5.y + loc_0.y;
    w.x = 1.00000000000000000e+00f;
    w.y = 0.00000000000000000e+00f;
    loc_0.x = temp_1.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_1.x, w.x, loc_0.x);
    loc_0.y = temp_1.y * w.x;
    loc_0.y = fma(temp_1.x, w.y, loc_0.y);
    temp_1.x = temp_0.x - loc_0.x;
    temp_1.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    iw.x = w.y;
    iw.y = -w.x;
    loc_0.x = temp_3.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_3.x, iw.x, loc_0.x);
    loc_0.y = temp_3.y * iw.x;
    loc_0.y = fma(temp_3.x, iw.y, loc_0.y);
    temp_3.x = temp_2.x - loc_0.x;
    temp_3.y = temp_2.y - loc_0.y;
    temp_2.x = temp_2.x + loc_0.x;
    temp_2.y = temp_2.y + loc_0.y;
    iw.x = w.y * -7.07106781186547573e-01f;
    iw.x = -iw.x;
    iw.x = fma(w.x, 7.07106781186547573e-01f, iw.x);
    iw.y = w.y * 7.07106781186547573e-01f;
    iw.y = fma(w.x, -7.07106781186547573e-01f, iw.y);
    loc_0.x = temp_5.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_5.x, iw.x, loc_0.x);
    loc_0.y = temp_5.y * iw.x;
    loc_0.y = fma(temp_5.x, iw.y, loc_0.y);
    temp_5.x = temp_4.x - loc_0.x;
    temp_5.y = temp_4.y - loc_0.y;
    temp_4.x = temp_4.x + loc_0.x;
    temp_4.y = temp_4.y + loc_0.y;
    w.x = iw.y;
    w.y = -iw.x;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_6.x - loc_0.x;
    temp_7.y = temp_6.y - loc_0.y;
    temp_6.x = temp_6.x + loc_0.x;
    temp_6.y = temp_6.y + loc_0.y;
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 1;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID = blockInvocationID * 8;
    inoutID = inoutID + stageInvocationID;
    sdataID = inoutID + 0;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_0;
    sdataID = inoutID + 1;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_4;
    sdataID = inoutID + 2;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_2;
    sdataID = inoutID + 3;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_6;
    sdataID = inoutID + 4;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_1;
    sdataID = inoutID + 5;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_5;
    sdataID = inoutID + 6;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_3;
    sdataID = inoutID + 7;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_7;
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    stageInvocationID = stageInvocationID % 8;
    angle = stageInvocationID * -3.92699081698724139e-01f;
    sdataID = gl_LocalInvocationID.x + 0;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_0 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 32;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_4 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 64;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_2 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 96;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_6 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 128;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_1 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 160;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_5 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 192;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_3 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 224;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_7 = sdata[sdataID];
    w.x = cos(angle);
    w.y = sin(angle);
    loc_0.x = temp_1.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_1.x, w.x, loc_0.x);
    loc_0.y = temp_1.y * w.x;
    loc_0.y = fma(temp_1.x, w.y, loc_0.y);
    temp_1.x = temp_0.x - loc_0.x;
    temp_1.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = temp_5.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_5.x, w.x, loc_0.x);
    loc_0.y = temp_5.y * w.x;
    loc_0.y = fma(temp_5.x, w.y, loc_0.y);
    temp_5.x = temp_4.x - loc_0.x;
    temp_5.y = temp_4.y - loc_0.y;
    temp_4.x = temp_4.x + loc_0.x;
    temp_4.y = temp_4.y + loc_0.y;
    loc_0.x = temp_3.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_3.x, w.x, loc_0.x);
    loc_0.y = temp_3.y * w.x;
    loc_0.y = fma(temp_3.x, w.y, loc_0.y);
    temp_3.x = temp_2.x - loc_0.x;
    temp_3.y = temp_2.y - loc_0.y;
    temp_2.x = temp_2.x + loc_0.x;
    temp_2.y = temp_2.y + loc_0.y;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_6.x - loc_0.x;
    temp_7.y = temp_6.y - loc_0.y;
    temp_6.x = temp_6.x + loc_0.x;
    temp_6.y = temp_6.y + loc_0.y;
    loc_0.x = angle * 5.00000000000000000e-01f;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);
    loc_0.x = temp_2.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_2.x, w.x, loc_0.x);
    loc_0.y = temp_2.y * w.x;
    loc_0.y = fma(temp_2.x, w.y, loc_0.y);
    temp_2.x = temp_0.x - loc_0.x;
    temp_2.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = temp_6.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, w.x, loc_0.x);
    loc_0.y = temp_6.y * w.x;
    loc_0.y = fma(temp_6.x, w.y, loc_0.y);
    temp_6.x = temp_4.x - loc_0.x;
    temp_6.y = temp_4.y - loc_0.y;
    temp_4.x = temp_4.x + loc_0.x;
    temp_4.y = temp_4.y + loc_0.y;
    iw.x = w.y;
    iw.y = -w.x;
    loc_0.x = temp_3.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_3.x, iw.x, loc_0.x);
    loc_0.y = temp_3.y * iw.x;
    loc_0.y = fma(temp_3.x, iw.y, loc_0.y);
    temp_3.x = temp_1.x - loc_0.x;
    temp_3.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    loc_0.x = temp_7.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, iw.x, loc_0.x);
    loc_0.y = temp_7.y * iw.x;
    loc_0.y = fma(temp_7.x, iw.y, loc_0.y);
    temp_7.x = temp_5.x - loc_0.x;
    temp_7.y = temp_5.y - loc_0.y;
    temp_5.x = temp_5.x + loc_0.x;
    temp_5.y = temp_5.y + loc_0.y;
    loc_0.x = angle * 2.50000000000000000e-01f;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);
    loc_0.x = temp_4.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_4.x, w.x, loc_0.x);
    loc_0.y = temp_4.y * w.x;
    loc_0.y = fma(temp_4.x, w.y, loc_0.y);
    temp_4.x = temp_0.x - loc_0.x;
    temp_4.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    iw.x = w.y;
    iw.y = -w.x;
    loc_0.x = temp_6.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, iw.x, loc_0.x);
    loc_0.y = temp_6.y * iw.x;
    loc_0.y = fma(temp_6.x, iw.y, loc_0.y);
    temp_6.x = temp_2.x - loc_0.x;
    temp_6.y = temp_2.y - loc_0.y;
    temp_2.x = temp_2.x + loc_0.x;
    temp_2.y = temp_2.y + loc_0.y;
    iw.x = w.y * -7.07106781186547573e-01f;
    iw.x = -iw.x;
    iw.x = fma(w.x, 7.07106781186547573e-01f, iw.x);
    iw.y = w.y * 7.07106781186547573e-01f;
    iw.y = fma(w.x, -7.07106781186547573e-01f, iw.y);
    loc_0.x = temp_5.y * iw.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_5.x, iw.x, loc_0.x);
    loc_0.y = temp_5.y * iw.x;
    loc_0.y = fma(temp_5.x, iw.y, loc_0.y);
    temp_5.x = temp_1.x - loc_0.x;
    temp_5.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    w.x = iw.y;
    w.y = -iw.x;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_3.x - loc_0.x;
    temp_7.y = temp_3.y - loc_0.y;
    temp_3.x = temp_3.x + loc_0.x;
    temp_3.y = temp_3.y + loc_0.y;
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 8;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID = blockInvocationID * 8;
    inoutID = inoutID + stageInvocationID;
    sdataID = inoutID + 0;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_0;
    sdataID = inoutID + 8;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_1;
    sdataID = inoutID + 16;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_2;
    sdataID = inoutID + 24;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_3;
    sdataID = inoutID + 32;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_4;
    sdataID = inoutID + 40;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_5;
    sdataID = inoutID + 48;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_6;
    sdataID = inoutID + 56;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    sdata[sdataID] = temp_7;
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    stageInvocationID = stageInvocationID % 64;
    angle = stageInvocationID * -4.90873852123405174e-02f;
    sdataID = gl_LocalInvocationID.x + 0;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_0 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 64;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_2 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 128;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_4 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 192;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_6 = sdata[sdataID];
    w.x = cos(angle);
    w.y = sin(angle);
    loc_0.x = temp_4.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_4.x, w.x, loc_0.x);
    loc_0.y = temp_4.y * w.x;
    loc_0.y = fma(temp_4.x, w.y, loc_0.y);
    temp_4.x = temp_0.x - loc_0.x;
    temp_4.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = temp_6.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, w.x, loc_0.x);
    loc_0.y = temp_6.y * w.x;
    loc_0.y = fma(temp_6.x, w.y, loc_0.y);
    temp_6.x = temp_2.x - loc_0.x;
    temp_6.y = temp_2.y - loc_0.y;
    temp_2.x = temp_2.x + loc_0.x;
    temp_2.y = temp_2.y + loc_0.y;
    loc_0.x = angle * 5.00000000000000000e-01f;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);
    loc_0.x = temp_2.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_2.x, w.x, loc_0.x);
    loc_0.y = temp_2.y * w.x;
    loc_0.y = fma(temp_2.x, w.y, loc_0.y);
    temp_2.x = temp_0.x - loc_0.x;
    temp_2.y = temp_0.y - loc_0.y;
    temp_0.x = temp_0.x + loc_0.x;
    temp_0.y = temp_0.y + loc_0.y;
    loc_0.x = w.x;
    w.x = w.y;
    w.y = -loc_0.x;
    loc_0.x = temp_6.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_6.x, w.x, loc_0.x);
    loc_0.y = temp_6.y * w.x;
    loc_0.y = fma(temp_6.x, w.y, loc_0.y);
    temp_6.x = temp_4.x - loc_0.x;
    temp_6.y = temp_4.y - loc_0.y;
    temp_4.x = temp_4.x + loc_0.x;
    temp_4.y = temp_4.y + loc_0.y;
    stageInvocationID = gl_LocalInvocationID.x + 32;
    stageInvocationID = stageInvocationID % 64;
    angle = stageInvocationID * -4.90873852123405174e-02f;
    sdataID = gl_LocalInvocationID.x + 32;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_1 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 96;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_3 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 160;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_5 = sdata[sdataID];
    sdataID = gl_LocalInvocationID.x + 224;
    tempInt = sdataID / 16;
    tempInt = tempInt * 17;
    sdataID = sdataID % 16;
    sdataID = sdataID + tempInt;
    temp_7 = sdata[sdataID];
    w.x = cos(angle);
    w.y = sin(angle);
    loc_0.x = temp_5.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_5.x, w.x, loc_0.x);
    loc_0.y = temp_5.y * w.x;
    loc_0.y = fma(temp_5.x, w.y, loc_0.y);
    temp_5.x = temp_1.x - loc_0.x;
    temp_5.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_3.x - loc_0.x;
    temp_7.y = temp_3.y - loc_0.y;
    temp_3.x = temp_3.x + loc_0.x;
    temp_3.y = temp_3.y + loc_0.y;
    loc_0.x = angle * 5.00000000000000000e-01f;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);
    loc_0.x = temp_3.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_3.x, w.x, loc_0.x);
    loc_0.y = temp_3.y * w.x;
    loc_0.y = fma(temp_3.x, w.y, loc_0.y);
    temp_3.x = temp_1.x - loc_0.x;
    temp_3.y = temp_1.y - loc_0.y;
    temp_1.x = temp_1.x + loc_0.x;
    temp_1.y = temp_1.y + loc_0.y;
    loc_0.x = w.x;
    w.x = w.y;
    w.y = -loc_0.x;
    loc_0.x = temp_7.y * w.y;
    loc_0.x = -loc_0.x;
    loc_0.x = fma(temp_7.x, w.x, loc_0.x);
    loc_0.y = temp_7.y * w.x;
    loc_0.y = fma(temp_7.x, w.y, loc_0.y);
    temp_7.x = temp_5.x - loc_0.x;
    temp_7.y = temp_5.y - loc_0.y;
    temp_5.x = temp_5.x + loc_0.x;
    temp_5.y = temp_5.y + loc_0.y;
    inoutID = 256 * shiftY;
    inoutID = inoutID + shiftZ;
    inoutID = inoutID + gl_LocalInvocationID.x;
    combinedID = gl_LocalInvocationID.x + 0;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    outputs[inoutID] = temp_0;
    combinedID = gl_LocalInvocationID.x + 32;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_1;
    combinedID = gl_LocalInvocationID.x + 64;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_4;
    combinedID = gl_LocalInvocationID.x + 96;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_5;
    combinedID = gl_LocalInvocationID.x + 128;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_2;
    combinedID = gl_LocalInvocationID.x + 160;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_3;
    combinedID = gl_LocalInvocationID.x + 192;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_6;
    combinedID = gl_LocalInvocationID.x + 224;
    inoutID_x = combinedID % 256;
    inoutID_y = combinedID / 256;
    inoutID_y = inoutID_y + shiftY;
    inoutID = inoutID + 32;
    outputs[inoutID] = temp_7;
}