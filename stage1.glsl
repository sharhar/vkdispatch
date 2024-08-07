#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_debug_printf : enable

layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer DataIn{
	vec2 inputs[65536];
};

layout(std430, binding = 1) buffer DataOut{
	vec2 outputs[65536];
};

shared vec2 sdata[272];// sharedStride - fft size,  gl_WorkGroupSize.y - grouped consecutive ffts
//shared vec2 sdata[4096];

vec2 do_complex_mult(vec2 a, vec2 b) {
    vec2 res;

    res.x = a.y * b.y;
    res.x = fma(a.x, b.x, -res.x);
    res.y = a.y * b.x;
    res.y = fma(a.x, b.y, res.y);

    return res;
}

uint get_sdata(uint index) {
    uint result = (index % 16) + ((index / 16) * 17);

    //if(result >= 272) {
        //result = -1;
    //    debugPrintfEXT("Error: get_sdata index out of bounds: %d", result);
   // }

    return result;
}

void main() {
    //debugPrintfEXT("Hello from shader");

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
    vec2 iw = vec2(0);
    uint stageInvocationID = 0;
    uint blockInvocationID = 0;
    uint sdataID = 0;
    uint inoutID = 0;
    float angle = 0;

    inoutID = 256 * gl_WorkGroupID.y;
    inoutID = inoutID + gl_LocalInvocationID.x;

    temp_0 = inputs[inoutID +   0];
    temp_1 = inputs[inoutID +  32];
    temp_2 = inputs[inoutID +  64];
    temp_3 = inputs[inoutID +  96];
    temp_4 = inputs[inoutID + 128];
    temp_5 = inputs[inoutID + 160];
    temp_6 = inputs[inoutID + 192];
    temp_7 = inputs[inoutID + 224];

    w = vec2(1, 0);
    
    loc_0 = do_complex_mult(temp_4, w);
    temp_4 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;

    loc_0 = do_complex_mult(temp_5, w);
    temp_5 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;

    loc_0 = do_complex_mult(temp_6, w);
    temp_6 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;

    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_3 - loc_0;
    temp_3 = temp_3 + loc_0;

    w = vec2(1, 0);

    loc_0 = do_complex_mult(temp_2, w);
    temp_2 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;

    loc_0 = do_complex_mult(temp_3, w);
    temp_3 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;

    iw.x = w.y;
    iw.y = -w.x;

    loc_0 = do_complex_mult(temp_6, iw);
    temp_6 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;

    loc_0 = do_complex_mult(temp_7, iw);
    temp_7 = temp_5 - loc_0;
    temp_5 = temp_5 + loc_0;

    w = vec2(1, 0);

    loc_0 = do_complex_mult(temp_1, w);
    temp_1 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;

    iw.x = w.y;
    iw.y = -w.x;
    
    loc_0 = do_complex_mult(temp_3, iw);
    temp_3 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;

    iw = do_complex_mult(w, vec2(1, -1) / sqrt(2));
    
    loc_0 = do_complex_mult(temp_5, iw);
    temp_5 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    
    w.x = iw.y;
    w.y = -iw.x;

    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_6 - loc_0;
    temp_6 = temp_6 + loc_0;
    
    memoryBarrier();
    barrier();

    stageInvocationID = 0;
    blockInvocationID = gl_LocalInvocationID.x;

    inoutID = blockInvocationID * 8;
    inoutID = inoutID + stageInvocationID;

    sdata[get_sdata(inoutID + 0)] = temp_0;
    sdata[get_sdata(inoutID + 1)] = temp_4;
    sdata[get_sdata(inoutID + 2)] = temp_2;
    sdata[get_sdata(inoutID + 3)] = temp_6;
    sdata[get_sdata(inoutID + 4)] = temp_1;
    sdata[get_sdata(inoutID + 5)] = temp_5;
    sdata[get_sdata(inoutID + 6)] = temp_3;
    sdata[get_sdata(inoutID + 7)] = temp_7;
    
    memoryBarrier();
    barrier();

    stageInvocationID = gl_LocalInvocationID.x % 8;
    angle = stageInvocationID * -3.92699081698724139e-01f; // stageInvocationID * -pi / 8

    temp_0 = sdata[get_sdata(gl_LocalInvocationID.x +   0)];
    temp_4 = sdata[get_sdata(gl_LocalInvocationID.x +  32)];
    temp_2 = sdata[get_sdata(gl_LocalInvocationID.x +  64)];
    temp_6 = sdata[get_sdata(gl_LocalInvocationID.x +  96)];
    temp_1 = sdata[get_sdata(gl_LocalInvocationID.x + 128)];
    temp_5 = sdata[get_sdata(gl_LocalInvocationID.x + 160)];
    temp_3 = sdata[get_sdata(gl_LocalInvocationID.x + 192)];
    temp_7 = sdata[get_sdata(gl_LocalInvocationID.x + 224)];
    
    w.x = cos(angle);
    w.y = sin(angle);

    loc_0 = do_complex_mult(temp_1, w);
    temp_1 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;

    loc_0 = do_complex_mult(temp_5, w);
    temp_5 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    
    loc_0 = do_complex_mult(temp_3, w);
    temp_3 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;
    
    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_6 - loc_0;
    temp_6 = temp_6 + loc_0;
    
    loc_0.x = angle * 0.5;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);

    loc_0 = do_complex_mult(temp_2, w);
    temp_2 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    
    loc_0 = do_complex_mult(temp_6, w);
    temp_6 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    
    iw.x = w.y;
    iw.y = -w.x;

    loc_0 = do_complex_mult(temp_3, iw);
    temp_3 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;
    
    loc_0 = do_complex_mult(temp_7, iw);
    temp_7 = temp_5 - loc_0;
    temp_5 = temp_5 + loc_0;
    
    loc_0.x = angle * 0.25;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);

    loc_0 = do_complex_mult(temp_4, w);
    temp_4 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    
    iw.x = w.y;
    iw.y = -w.x;
    
    loc_0 = do_complex_mult(temp_6, iw);
    temp_6 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;

    iw = do_complex_mult(w, vec2(1, -1) / sqrt(2));
    
    loc_0 = do_complex_mult(temp_5, iw);
    temp_5 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;
    
    w.x = iw.y;
    w.y = -iw.x;

    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_3 - loc_0;
    temp_3 = temp_3 + loc_0;
    
    memoryBarrier();
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    blockInvocationID = stageInvocationID;
    stageInvocationID = stageInvocationID % 8;
    blockInvocationID = blockInvocationID - stageInvocationID;
    inoutID = blockInvocationID * 8;
    inoutID = inoutID + stageInvocationID;

    sdata[get_sdata(inoutID +  0)] = temp_0;
    sdata[get_sdata(inoutID +  8)] = temp_1;
    sdata[get_sdata(inoutID + 16)] = temp_2;
    sdata[get_sdata(inoutID + 24)] = temp_3;
    sdata[get_sdata(inoutID + 32)] = temp_4;
    sdata[get_sdata(inoutID + 40)] = temp_5;
    sdata[get_sdata(inoutID + 48)] = temp_6;
    sdata[get_sdata(inoutID + 56)] = temp_7;

    memoryBarrier();
    barrier();

    stageInvocationID = gl_LocalInvocationID.x + 0;
    stageInvocationID = stageInvocationID % 64;
    angle = stageInvocationID * -4.90873852123405174e-02f; // stageInvocationID * -pi / 64

    temp_0 = sdata[get_sdata(gl_LocalInvocationID.x +   0)];
    temp_2 = sdata[get_sdata(gl_LocalInvocationID.x +  64)];
    temp_4 = sdata[get_sdata(gl_LocalInvocationID.x + 128)];
    temp_6 = sdata[get_sdata(gl_LocalInvocationID.x + 192)];

    w.x = cos(angle);
    w.y = sin(angle);
    
    loc_0 = do_complex_mult(temp_4, w);
    temp_4 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;
    
    loc_0 = do_complex_mult(temp_6, w);
    temp_6 = temp_2 - loc_0;
    temp_2 = temp_2 + loc_0;

    loc_0.x = angle * 0.5;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);
    
    loc_0 = do_complex_mult(temp_2, w);
    temp_2 = temp_0 - loc_0;
    temp_0 = temp_0 + loc_0;

    loc_0.x = w.x;
    w.x = w.y;
    w.y = -loc_0.x;
    
    loc_0 = do_complex_mult(temp_6, w);
    temp_6 = temp_4 - loc_0;
    temp_4 = temp_4 + loc_0;
    
    stageInvocationID = gl_LocalInvocationID.x + 32;
    stageInvocationID = stageInvocationID % 64;
    angle = stageInvocationID * -4.90873852123405174e-02f; // stageInvocationID * -pi / 64

    temp_1 = sdata[get_sdata(gl_LocalInvocationID.x +  32)];
    temp_3 = sdata[get_sdata(gl_LocalInvocationID.x +  96)];
    temp_5 = sdata[get_sdata(gl_LocalInvocationID.x + 160)];
    temp_7 = sdata[get_sdata(gl_LocalInvocationID.x + 224)];
    
    w.x = cos(angle);
    w.y = sin(angle);
    
    loc_0 = do_complex_mult(temp_5, w);
    temp_5 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;

    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_3 - loc_0;
    temp_3 = temp_3 + loc_0;

    loc_0.x = angle * 0.5;
    w.x = cos(loc_0.x);
    w.y = sin(loc_0.x);

    loc_0 = do_complex_mult(temp_3, w);
    temp_3 = temp_1 - loc_0;
    temp_1 = temp_1 + loc_0;

    loc_0.x = w.x;
    w.x = w.y;
    w.y = -loc_0.x;

    loc_0 = do_complex_mult(temp_7, w);
    temp_7 = temp_5 - loc_0;
    temp_5 = temp_5 + loc_0;
    
    inoutID = 256 * gl_WorkGroupID.y;
    inoutID = inoutID + gl_LocalInvocationID.x;

    outputs[inoutID +   0] = temp_0;
    outputs[inoutID +  32] = temp_1;
    outputs[inoutID +  64] = temp_4;
    outputs[inoutID +  96] = temp_5;
    outputs[inoutID + 128] = temp_2;
    outputs[inoutID + 160] = temp_3;
    outputs[inoutID + 192] = temp_6;
    outputs[inoutID + 224] = temp_7;
}