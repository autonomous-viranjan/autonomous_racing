/*
 *    This file was auto-generated using the ACADO Toolkit.
 *    
 *    While ACADO Toolkit is free software released under the terms of
 *    the GNU Lesser General Public License (LGPL), the generated code
 *    as such remains the property of the user who used ACADO Toolkit
 *    to generate this code. In particular, user dependent data of the code
 *    do not inherit the GNU LGPL license. On the other hand, parts of the
 *    generated code that are a direct copy of source code from the
 *    ACADO Toolkit or the software tools it is based on, remain, as derived
 *    work, automatically covered by the LGPL license.
 *    
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *    
 */


#include "acado_common.h"




/******************************************************************************/
/*                                                                            */
/* ACADO code generation                                                      */
/*                                                                            */
/******************************************************************************/


int acado_modelSimulation(  )
{
int ret;

int lRun1;
ret = 0;
for (lRun1 = 0; lRun1 < 5; ++lRun1)
{
acadoWorkspace.state[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.state[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[lRun1 * 4 + 3];

acadoWorkspace.state[28] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.state[29] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.state[30] = acadoVariables.od[lRun1 * 6];
acadoWorkspace.state[31] = acadoVariables.od[lRun1 * 6 + 1];
acadoWorkspace.state[32] = acadoVariables.od[lRun1 * 6 + 2];
acadoWorkspace.state[33] = acadoVariables.od[lRun1 * 6 + 3];
acadoWorkspace.state[34] = acadoVariables.od[lRun1 * 6 + 4];
acadoWorkspace.state[35] = acadoVariables.od[lRun1 * 6 + 5];

ret = acado_integrate(acadoWorkspace.state, 1);

acadoWorkspace.d[lRun1 * 4] = acadoWorkspace.state[0] - acadoVariables.x[lRun1 * 4 + 4];
acadoWorkspace.d[lRun1 * 4 + 1] = acadoWorkspace.state[1] - acadoVariables.x[lRun1 * 4 + 5];
acadoWorkspace.d[lRun1 * 4 + 2] = acadoWorkspace.state[2] - acadoVariables.x[lRun1 * 4 + 6];
acadoWorkspace.d[lRun1 * 4 + 3] = acadoWorkspace.state[3] - acadoVariables.x[lRun1 * 4 + 7];

acadoWorkspace.evGx[lRun1 * 16] = acadoWorkspace.state[4];
acadoWorkspace.evGx[lRun1 * 16 + 1] = acadoWorkspace.state[5];
acadoWorkspace.evGx[lRun1 * 16 + 2] = acadoWorkspace.state[6];
acadoWorkspace.evGx[lRun1 * 16 + 3] = acadoWorkspace.state[7];
acadoWorkspace.evGx[lRun1 * 16 + 4] = acadoWorkspace.state[8];
acadoWorkspace.evGx[lRun1 * 16 + 5] = acadoWorkspace.state[9];
acadoWorkspace.evGx[lRun1 * 16 + 6] = acadoWorkspace.state[10];
acadoWorkspace.evGx[lRun1 * 16 + 7] = acadoWorkspace.state[11];
acadoWorkspace.evGx[lRun1 * 16 + 8] = acadoWorkspace.state[12];
acadoWorkspace.evGx[lRun1 * 16 + 9] = acadoWorkspace.state[13];
acadoWorkspace.evGx[lRun1 * 16 + 10] = acadoWorkspace.state[14];
acadoWorkspace.evGx[lRun1 * 16 + 11] = acadoWorkspace.state[15];
acadoWorkspace.evGx[lRun1 * 16 + 12] = acadoWorkspace.state[16];
acadoWorkspace.evGx[lRun1 * 16 + 13] = acadoWorkspace.state[17];
acadoWorkspace.evGx[lRun1 * 16 + 14] = acadoWorkspace.state[18];
acadoWorkspace.evGx[lRun1 * 16 + 15] = acadoWorkspace.state[19];

acadoWorkspace.evGu[lRun1 * 8] = acadoWorkspace.state[20];
acadoWorkspace.evGu[lRun1 * 8 + 1] = acadoWorkspace.state[21];
acadoWorkspace.evGu[lRun1 * 8 + 2] = acadoWorkspace.state[22];
acadoWorkspace.evGu[lRun1 * 8 + 3] = acadoWorkspace.state[23];
acadoWorkspace.evGu[lRun1 * 8 + 4] = acadoWorkspace.state[24];
acadoWorkspace.evGu[lRun1 * 8 + 5] = acadoWorkspace.state[25];
acadoWorkspace.evGu[lRun1 * 8 + 6] = acadoWorkspace.state[26];
acadoWorkspace.evGu[lRun1 * 8 + 7] = acadoWorkspace.state[27];
}
return ret;
}

void acado_evaluateLSQ(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 4;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[2];
out[3] = xd[3];
out[4] = u[0];
out[5] = u[1];
}

void acado_evaluateLSQEndTerm(const real_t* in, real_t* out)
{
const real_t* xd = in;

/* Compute outputs: */
out[0] = xd[0];
out[1] = xd[1];
out[2] = xd[3];
out[3] = xd[2];
}

void acado_evaluatePathConstraints(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* od = in + 6;
/* Vector of auxiliary variables; number of elements: 18. */
real_t* a = acadoWorkspace.conAuxVar;

/* Compute intermediate quantities: */
a[0] = (pow((((real_t)(0.0000000000000000e+00)-od[0])+xd[0]),2));
a[1] = (pow((((real_t)(0.0000000000000000e+00)-od[1])+xd[1]),2));
a[2] = ((od[2])*(od[2]));
a[3] = (pow((((real_t)(0.0000000000000000e+00)-od[3])+xd[0]),2));
a[4] = (pow((((real_t)(0.0000000000000000e+00)-od[4])+xd[1]),2));
a[5] = ((od[5])*(od[5]));
a[6] = ((real_t)(2.0000000000000000e+00)*(((real_t)(0.0000000000000000e+00)-od[0])+xd[0]));
a[7] = ((real_t)(2.0000000000000000e+00)*(((real_t)(0.0000000000000000e+00)-od[1])+xd[1]));
a[8] = (real_t)(0.0000000000000000e+00);
a[9] = (real_t)(0.0000000000000000e+00);
a[10] = ((real_t)(2.0000000000000000e+00)*(((real_t)(0.0000000000000000e+00)-od[3])+xd[0]));
a[11] = ((real_t)(2.0000000000000000e+00)*(((real_t)(0.0000000000000000e+00)-od[4])+xd[1]));
a[12] = (real_t)(0.0000000000000000e+00);
a[13] = (real_t)(0.0000000000000000e+00);
a[14] = (real_t)(0.0000000000000000e+00);
a[15] = (real_t)(0.0000000000000000e+00);
a[16] = (real_t)(0.0000000000000000e+00);
a[17] = (real_t)(0.0000000000000000e+00);

/* Compute outputs: */
out[0] = ((a[0]+a[1])-a[2]);
out[1] = ((a[3]+a[4])-a[5]);
out[2] = a[6];
out[3] = a[7];
out[4] = a[8];
out[5] = a[9];
out[6] = a[10];
out[7] = a[11];
out[8] = a[12];
out[9] = a[13];
out[10] = a[14];
out[11] = a[15];
out[12] = a[16];
out[13] = a[17];
}

void acado_setObjQ1Q2( real_t* const tmpObjS, real_t* const tmpQ1, real_t* const tmpQ2 )
{
tmpQ2[0] = +tmpObjS[0];
tmpQ2[1] = +tmpObjS[1];
tmpQ2[2] = +tmpObjS[2];
tmpQ2[3] = +tmpObjS[3];
tmpQ2[4] = +tmpObjS[4];
tmpQ2[5] = +tmpObjS[5];
tmpQ2[6] = +tmpObjS[6];
tmpQ2[7] = +tmpObjS[7];
tmpQ2[8] = +tmpObjS[8];
tmpQ2[9] = +tmpObjS[9];
tmpQ2[10] = +tmpObjS[10];
tmpQ2[11] = +tmpObjS[11];
tmpQ2[12] = +tmpObjS[12];
tmpQ2[13] = +tmpObjS[13];
tmpQ2[14] = +tmpObjS[14];
tmpQ2[15] = +tmpObjS[15];
tmpQ2[16] = +tmpObjS[16];
tmpQ2[17] = +tmpObjS[17];
tmpQ2[18] = +tmpObjS[18];
tmpQ2[19] = +tmpObjS[19];
tmpQ2[20] = +tmpObjS[20];
tmpQ2[21] = +tmpObjS[21];
tmpQ2[22] = +tmpObjS[22];
tmpQ2[23] = +tmpObjS[23];
tmpQ1[0] = + tmpQ2[0];
tmpQ1[1] = + tmpQ2[1];
tmpQ1[2] = + tmpQ2[2];
tmpQ1[3] = + tmpQ2[3];
tmpQ1[4] = + tmpQ2[6];
tmpQ1[5] = + tmpQ2[7];
tmpQ1[6] = + tmpQ2[8];
tmpQ1[7] = + tmpQ2[9];
tmpQ1[8] = + tmpQ2[12];
tmpQ1[9] = + tmpQ2[13];
tmpQ1[10] = + tmpQ2[14];
tmpQ1[11] = + tmpQ2[15];
tmpQ1[12] = + tmpQ2[18];
tmpQ1[13] = + tmpQ2[19];
tmpQ1[14] = + tmpQ2[20];
tmpQ1[15] = + tmpQ2[21];
}

void acado_setObjR1R2( real_t* const tmpObjS, real_t* const tmpR1, real_t* const tmpR2 )
{
tmpR2[0] = +tmpObjS[24];
tmpR2[1] = +tmpObjS[25];
tmpR2[2] = +tmpObjS[26];
tmpR2[3] = +tmpObjS[27];
tmpR2[4] = +tmpObjS[28];
tmpR2[5] = +tmpObjS[29];
tmpR2[6] = +tmpObjS[30];
tmpR2[7] = +tmpObjS[31];
tmpR2[8] = +tmpObjS[32];
tmpR2[9] = +tmpObjS[33];
tmpR2[10] = +tmpObjS[34];
tmpR2[11] = +tmpObjS[35];
tmpR1[0] = + tmpR2[4];
tmpR1[1] = + tmpR2[5];
tmpR1[2] = + tmpR2[10];
tmpR1[3] = + tmpR2[11];
}

void acado_setObjQN1QN2( real_t* const tmpObjSEndTerm, real_t* const tmpQN1, real_t* const tmpQN2 )
{
tmpQN2[0] = +tmpObjSEndTerm[0];
tmpQN2[1] = +tmpObjSEndTerm[1];
tmpQN2[2] = +tmpObjSEndTerm[2];
tmpQN2[3] = +tmpObjSEndTerm[3];
tmpQN2[4] = +tmpObjSEndTerm[4];
tmpQN2[5] = +tmpObjSEndTerm[5];
tmpQN2[6] = +tmpObjSEndTerm[6];
tmpQN2[7] = +tmpObjSEndTerm[7];
tmpQN2[8] = +tmpObjSEndTerm[12];
tmpQN2[9] = +tmpObjSEndTerm[13];
tmpQN2[10] = +tmpObjSEndTerm[14];
tmpQN2[11] = +tmpObjSEndTerm[15];
tmpQN2[12] = +tmpObjSEndTerm[8];
tmpQN2[13] = +tmpObjSEndTerm[9];
tmpQN2[14] = +tmpObjSEndTerm[10];
tmpQN2[15] = +tmpObjSEndTerm[11];
tmpQN1[0] = + tmpQN2[0];
tmpQN1[1] = + tmpQN2[1];
tmpQN1[2] = + tmpQN2[3];
tmpQN1[3] = + tmpQN2[2];
tmpQN1[4] = + tmpQN2[4];
tmpQN1[5] = + tmpQN2[5];
tmpQN1[6] = + tmpQN2[7];
tmpQN1[7] = + tmpQN2[6];
tmpQN1[8] = + tmpQN2[8];
tmpQN1[9] = + tmpQN2[9];
tmpQN1[10] = + tmpQN2[11];
tmpQN1[11] = + tmpQN2[10];
tmpQN1[12] = + tmpQN2[12];
tmpQN1[13] = + tmpQN2[13];
tmpQN1[14] = + tmpQN2[15];
tmpQN1[15] = + tmpQN2[14];
}

void acado_evaluateObjective(  )
{
int runObj;
for (runObj = 0; runObj < 5; ++runObj)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[runObj * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[runObj * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[runObj * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[runObj * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[runObj * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[runObj * 2 + 1];
acadoWorkspace.objValueIn[6] = acadoVariables.od[runObj * 6];
acadoWorkspace.objValueIn[7] = acadoVariables.od[runObj * 6 + 1];
acadoWorkspace.objValueIn[8] = acadoVariables.od[runObj * 6 + 2];
acadoWorkspace.objValueIn[9] = acadoVariables.od[runObj * 6 + 3];
acadoWorkspace.objValueIn[10] = acadoVariables.od[runObj * 6 + 4];
acadoWorkspace.objValueIn[11] = acadoVariables.od[runObj * 6 + 5];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[runObj * 6] = acadoWorkspace.objValueOut[0];
acadoWorkspace.Dy[runObj * 6 + 1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.Dy[runObj * 6 + 2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.Dy[runObj * 6 + 3] = acadoWorkspace.objValueOut[3];
acadoWorkspace.Dy[runObj * 6 + 4] = acadoWorkspace.objValueOut[4];
acadoWorkspace.Dy[runObj * 6 + 5] = acadoWorkspace.objValueOut[5];

acado_setObjQ1Q2( acadoVariables.W, &(acadoWorkspace.Q1[ runObj * 16 ]), &(acadoWorkspace.Q2[ runObj * 24 ]) );

acado_setObjR1R2( acadoVariables.W, &(acadoWorkspace.R1[ runObj * 4 ]), &(acadoWorkspace.R2[ runObj * 12 ]) );

}
acadoWorkspace.objValueIn[0] = acadoVariables.x[20];
acadoWorkspace.objValueIn[1] = acadoVariables.x[21];
acadoWorkspace.objValueIn[2] = acadoVariables.x[22];
acadoWorkspace.objValueIn[3] = acadoVariables.x[23];
acadoWorkspace.objValueIn[4] = acadoVariables.od[30];
acadoWorkspace.objValueIn[5] = acadoVariables.od[31];
acadoWorkspace.objValueIn[6] = acadoVariables.od[32];
acadoWorkspace.objValueIn[7] = acadoVariables.od[33];
acadoWorkspace.objValueIn[8] = acadoVariables.od[34];
acadoWorkspace.objValueIn[9] = acadoVariables.od[35];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );

acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3];

acado_setObjQN1QN2( acadoVariables.WN, acadoWorkspace.QN1, acadoWorkspace.QN2 );

}

void acado_moveGxT( real_t* const Gx1, real_t* const Gx2 )
{
Gx2[0] = Gx1[0];
Gx2[1] = Gx1[1];
Gx2[2] = Gx1[2];
Gx2[3] = Gx1[3];
Gx2[4] = Gx1[4];
Gx2[5] = Gx1[5];
Gx2[6] = Gx1[6];
Gx2[7] = Gx1[7];
Gx2[8] = Gx1[8];
Gx2[9] = Gx1[9];
Gx2[10] = Gx1[10];
Gx2[11] = Gx1[11];
Gx2[12] = Gx1[12];
Gx2[13] = Gx1[13];
Gx2[14] = Gx1[14];
Gx2[15] = Gx1[15];
}

void acado_multGxGx( real_t* const Gx1, real_t* const Gx2, real_t* const Gx3 )
{
Gx3[0] = + Gx1[0]*Gx2[0] + Gx1[1]*Gx2[4] + Gx1[2]*Gx2[8] + Gx1[3]*Gx2[12];
Gx3[1] = + Gx1[0]*Gx2[1] + Gx1[1]*Gx2[5] + Gx1[2]*Gx2[9] + Gx1[3]*Gx2[13];
Gx3[2] = + Gx1[0]*Gx2[2] + Gx1[1]*Gx2[6] + Gx1[2]*Gx2[10] + Gx1[3]*Gx2[14];
Gx3[3] = + Gx1[0]*Gx2[3] + Gx1[1]*Gx2[7] + Gx1[2]*Gx2[11] + Gx1[3]*Gx2[15];
Gx3[4] = + Gx1[4]*Gx2[0] + Gx1[5]*Gx2[4] + Gx1[6]*Gx2[8] + Gx1[7]*Gx2[12];
Gx3[5] = + Gx1[4]*Gx2[1] + Gx1[5]*Gx2[5] + Gx1[6]*Gx2[9] + Gx1[7]*Gx2[13];
Gx3[6] = + Gx1[4]*Gx2[2] + Gx1[5]*Gx2[6] + Gx1[6]*Gx2[10] + Gx1[7]*Gx2[14];
Gx3[7] = + Gx1[4]*Gx2[3] + Gx1[5]*Gx2[7] + Gx1[6]*Gx2[11] + Gx1[7]*Gx2[15];
Gx3[8] = + Gx1[8]*Gx2[0] + Gx1[9]*Gx2[4] + Gx1[10]*Gx2[8] + Gx1[11]*Gx2[12];
Gx3[9] = + Gx1[8]*Gx2[1] + Gx1[9]*Gx2[5] + Gx1[10]*Gx2[9] + Gx1[11]*Gx2[13];
Gx3[10] = + Gx1[8]*Gx2[2] + Gx1[9]*Gx2[6] + Gx1[10]*Gx2[10] + Gx1[11]*Gx2[14];
Gx3[11] = + Gx1[8]*Gx2[3] + Gx1[9]*Gx2[7] + Gx1[10]*Gx2[11] + Gx1[11]*Gx2[15];
Gx3[12] = + Gx1[12]*Gx2[0] + Gx1[13]*Gx2[4] + Gx1[14]*Gx2[8] + Gx1[15]*Gx2[12];
Gx3[13] = + Gx1[12]*Gx2[1] + Gx1[13]*Gx2[5] + Gx1[14]*Gx2[9] + Gx1[15]*Gx2[13];
Gx3[14] = + Gx1[12]*Gx2[2] + Gx1[13]*Gx2[6] + Gx1[14]*Gx2[10] + Gx1[15]*Gx2[14];
Gx3[15] = + Gx1[12]*Gx2[3] + Gx1[13]*Gx2[7] + Gx1[14]*Gx2[11] + Gx1[15]*Gx2[15];
}

void acado_multGxGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[1]*Gu1[2] + Gx1[2]*Gu1[4] + Gx1[3]*Gu1[6];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[1]*Gu1[3] + Gx1[2]*Gu1[5] + Gx1[3]*Gu1[7];
Gu2[2] = + Gx1[4]*Gu1[0] + Gx1[5]*Gu1[2] + Gx1[6]*Gu1[4] + Gx1[7]*Gu1[6];
Gu2[3] = + Gx1[4]*Gu1[1] + Gx1[5]*Gu1[3] + Gx1[6]*Gu1[5] + Gx1[7]*Gu1[7];
Gu2[4] = + Gx1[8]*Gu1[0] + Gx1[9]*Gu1[2] + Gx1[10]*Gu1[4] + Gx1[11]*Gu1[6];
Gu2[5] = + Gx1[8]*Gu1[1] + Gx1[9]*Gu1[3] + Gx1[10]*Gu1[5] + Gx1[11]*Gu1[7];
Gu2[6] = + Gx1[12]*Gu1[0] + Gx1[13]*Gu1[2] + Gx1[14]*Gu1[4] + Gx1[15]*Gu1[6];
Gu2[7] = + Gx1[12]*Gu1[1] + Gx1[13]*Gu1[3] + Gx1[14]*Gu1[5] + Gx1[15]*Gu1[7];
}

void acado_moveGuE( real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = Gu1[0];
Gu2[1] = Gu1[1];
Gu2[2] = Gu1[2];
Gu2[3] = Gu1[3];
Gu2[4] = Gu1[4];
Gu2[5] = Gu1[5];
Gu2[6] = Gu1[6];
Gu2[7] = Gu1[7];
}

void acado_multBTW1( real_t* const Gu1, real_t* const Gu2, int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 20) + (iCol * 2)] = + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6];
acadoWorkspace.H[(iRow * 20) + (iCol * 2 + 1)] = + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7];
acadoWorkspace.H[(iRow * 20 + 10) + (iCol * 2)] = + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6];
acadoWorkspace.H[(iRow * 20 + 10) + (iCol * 2 + 1)] = + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7];
}

void acado_multBTW1_R1( real_t* const R11, real_t* const Gu1, real_t* const Gu2, int iRow )
{
acadoWorkspace.H[iRow * 22] = + Gu1[0]*Gu2[0] + Gu1[2]*Gu2[2] + Gu1[4]*Gu2[4] + Gu1[6]*Gu2[6] + R11[0];
acadoWorkspace.H[iRow * 22 + 1] = + Gu1[0]*Gu2[1] + Gu1[2]*Gu2[3] + Gu1[4]*Gu2[5] + Gu1[6]*Gu2[7] + R11[1];
acadoWorkspace.H[iRow * 22 + 10] = + Gu1[1]*Gu2[0] + Gu1[3]*Gu2[2] + Gu1[5]*Gu2[4] + Gu1[7]*Gu2[6] + R11[2];
acadoWorkspace.H[iRow * 22 + 11] = + Gu1[1]*Gu2[1] + Gu1[3]*Gu2[3] + Gu1[5]*Gu2[5] + Gu1[7]*Gu2[7] + R11[3];
acadoWorkspace.H[iRow * 22] += 1.0000000000000000e-10;
acadoWorkspace.H[iRow * 22 + 11] += 1.0000000000000000e-10;
}

void acado_multGxTGu( real_t* const Gx1, real_t* const Gu1, real_t* const Gu2 )
{
Gu2[0] = + Gx1[0]*Gu1[0] + Gx1[4]*Gu1[2] + Gx1[8]*Gu1[4] + Gx1[12]*Gu1[6];
Gu2[1] = + Gx1[0]*Gu1[1] + Gx1[4]*Gu1[3] + Gx1[8]*Gu1[5] + Gx1[12]*Gu1[7];
Gu2[2] = + Gx1[1]*Gu1[0] + Gx1[5]*Gu1[2] + Gx1[9]*Gu1[4] + Gx1[13]*Gu1[6];
Gu2[3] = + Gx1[1]*Gu1[1] + Gx1[5]*Gu1[3] + Gx1[9]*Gu1[5] + Gx1[13]*Gu1[7];
Gu2[4] = + Gx1[2]*Gu1[0] + Gx1[6]*Gu1[2] + Gx1[10]*Gu1[4] + Gx1[14]*Gu1[6];
Gu2[5] = + Gx1[2]*Gu1[1] + Gx1[6]*Gu1[3] + Gx1[10]*Gu1[5] + Gx1[14]*Gu1[7];
Gu2[6] = + Gx1[3]*Gu1[0] + Gx1[7]*Gu1[2] + Gx1[11]*Gu1[4] + Gx1[15]*Gu1[6];
Gu2[7] = + Gx1[3]*Gu1[1] + Gx1[7]*Gu1[3] + Gx1[11]*Gu1[5] + Gx1[15]*Gu1[7];
}

void acado_multQEW2( real_t* const Q11, real_t* const Gu1, real_t* const Gu2, real_t* const Gu3 )
{
Gu3[0] = + Q11[0]*Gu1[0] + Q11[1]*Gu1[2] + Q11[2]*Gu1[4] + Q11[3]*Gu1[6] + Gu2[0];
Gu3[1] = + Q11[0]*Gu1[1] + Q11[1]*Gu1[3] + Q11[2]*Gu1[5] + Q11[3]*Gu1[7] + Gu2[1];
Gu3[2] = + Q11[4]*Gu1[0] + Q11[5]*Gu1[2] + Q11[6]*Gu1[4] + Q11[7]*Gu1[6] + Gu2[2];
Gu3[3] = + Q11[4]*Gu1[1] + Q11[5]*Gu1[3] + Q11[6]*Gu1[5] + Q11[7]*Gu1[7] + Gu2[3];
Gu3[4] = + Q11[8]*Gu1[0] + Q11[9]*Gu1[2] + Q11[10]*Gu1[4] + Q11[11]*Gu1[6] + Gu2[4];
Gu3[5] = + Q11[8]*Gu1[1] + Q11[9]*Gu1[3] + Q11[10]*Gu1[5] + Q11[11]*Gu1[7] + Gu2[5];
Gu3[6] = + Q11[12]*Gu1[0] + Q11[13]*Gu1[2] + Q11[14]*Gu1[4] + Q11[15]*Gu1[6] + Gu2[6];
Gu3[7] = + Q11[12]*Gu1[1] + Q11[13]*Gu1[3] + Q11[14]*Gu1[5] + Q11[15]*Gu1[7] + Gu2[7];
}

void acado_macATw1QDy( real_t* const Gx1, real_t* const w11, real_t* const w12, real_t* const w13 )
{
w13[0] = + Gx1[0]*w11[0] + Gx1[4]*w11[1] + Gx1[8]*w11[2] + Gx1[12]*w11[3] + w12[0];
w13[1] = + Gx1[1]*w11[0] + Gx1[5]*w11[1] + Gx1[9]*w11[2] + Gx1[13]*w11[3] + w12[1];
w13[2] = + Gx1[2]*w11[0] + Gx1[6]*w11[1] + Gx1[10]*w11[2] + Gx1[14]*w11[3] + w12[2];
w13[3] = + Gx1[3]*w11[0] + Gx1[7]*w11[1] + Gx1[11]*w11[2] + Gx1[15]*w11[3] + w12[3];
}

void acado_macBTw1( real_t* const Gu1, real_t* const w11, real_t* const U1 )
{
U1[0] += + Gu1[0]*w11[0] + Gu1[2]*w11[1] + Gu1[4]*w11[2] + Gu1[6]*w11[3];
U1[1] += + Gu1[1]*w11[0] + Gu1[3]*w11[1] + Gu1[5]*w11[2] + Gu1[7]*w11[3];
}

void acado_macQSbarW2( real_t* const Q11, real_t* const w11, real_t* const w12, real_t* const w13 )
{
w13[0] = + Q11[0]*w11[0] + Q11[1]*w11[1] + Q11[2]*w11[2] + Q11[3]*w11[3] + w12[0];
w13[1] = + Q11[4]*w11[0] + Q11[5]*w11[1] + Q11[6]*w11[2] + Q11[7]*w11[3] + w12[1];
w13[2] = + Q11[8]*w11[0] + Q11[9]*w11[1] + Q11[10]*w11[2] + Q11[11]*w11[3] + w12[2];
w13[3] = + Q11[12]*w11[0] + Q11[13]*w11[1] + Q11[14]*w11[2] + Q11[15]*w11[3] + w12[3];
}

void acado_macASbar( real_t* const Gx1, real_t* const w11, real_t* const w12 )
{
w12[0] += + Gx1[0]*w11[0] + Gx1[1]*w11[1] + Gx1[2]*w11[2] + Gx1[3]*w11[3];
w12[1] += + Gx1[4]*w11[0] + Gx1[5]*w11[1] + Gx1[6]*w11[2] + Gx1[7]*w11[3];
w12[2] += + Gx1[8]*w11[0] + Gx1[9]*w11[1] + Gx1[10]*w11[2] + Gx1[11]*w11[3];
w12[3] += + Gx1[12]*w11[0] + Gx1[13]*w11[1] + Gx1[14]*w11[2] + Gx1[15]*w11[3];
}

void acado_expansionStep( real_t* const Gx1, real_t* const Gu1, real_t* const U1, real_t* const w11, real_t* const w12 )
{
w12[0] += + Gx1[0]*w11[0] + Gx1[1]*w11[1] + Gx1[2]*w11[2] + Gx1[3]*w11[3];
w12[1] += + Gx1[4]*w11[0] + Gx1[5]*w11[1] + Gx1[6]*w11[2] + Gx1[7]*w11[3];
w12[2] += + Gx1[8]*w11[0] + Gx1[9]*w11[1] + Gx1[10]*w11[2] + Gx1[11]*w11[3];
w12[3] += + Gx1[12]*w11[0] + Gx1[13]*w11[1] + Gx1[14]*w11[2] + Gx1[15]*w11[3];
w12[0] += + Gu1[0]*U1[0] + Gu1[1]*U1[1];
w12[1] += + Gu1[2]*U1[0] + Gu1[3]*U1[1];
w12[2] += + Gu1[4]*U1[0] + Gu1[5]*U1[1];
w12[3] += + Gu1[6]*U1[0] + Gu1[7]*U1[1];
}

void acado_copyHTH( int iRow, int iCol )
{
acadoWorkspace.H[(iRow * 20) + (iCol * 2)] = acadoWorkspace.H[(iCol * 20) + (iRow * 2)];
acadoWorkspace.H[(iRow * 20) + (iCol * 2 + 1)] = acadoWorkspace.H[(iCol * 20 + 10) + (iRow * 2)];
acadoWorkspace.H[(iRow * 20 + 10) + (iCol * 2)] = acadoWorkspace.H[(iCol * 20) + (iRow * 2 + 1)];
acadoWorkspace.H[(iRow * 20 + 10) + (iCol * 2 + 1)] = acadoWorkspace.H[(iCol * 20 + 10) + (iRow * 2 + 1)];
}

void acado_multRDy( real_t* const R2, real_t* const Dy1, real_t* const RDy1 )
{
RDy1[0] = + R2[0]*Dy1[0] + R2[1]*Dy1[1] + R2[2]*Dy1[2] + R2[3]*Dy1[3] + R2[4]*Dy1[4] + R2[5]*Dy1[5];
RDy1[1] = + R2[6]*Dy1[0] + R2[7]*Dy1[1] + R2[8]*Dy1[2] + R2[9]*Dy1[3] + R2[10]*Dy1[4] + R2[11]*Dy1[5];
}

void acado_multQDy( real_t* const Q2, real_t* const Dy1, real_t* const QDy1 )
{
QDy1[0] = + Q2[0]*Dy1[0] + Q2[1]*Dy1[1] + Q2[2]*Dy1[2] + Q2[3]*Dy1[3] + Q2[4]*Dy1[4] + Q2[5]*Dy1[5];
QDy1[1] = + Q2[6]*Dy1[0] + Q2[7]*Dy1[1] + Q2[8]*Dy1[2] + Q2[9]*Dy1[3] + Q2[10]*Dy1[4] + Q2[11]*Dy1[5];
QDy1[2] = + Q2[12]*Dy1[0] + Q2[13]*Dy1[1] + Q2[14]*Dy1[2] + Q2[15]*Dy1[3] + Q2[16]*Dy1[4] + Q2[17]*Dy1[5];
QDy1[3] = + Q2[18]*Dy1[0] + Q2[19]*Dy1[1] + Q2[20]*Dy1[2] + Q2[21]*Dy1[3] + Q2[22]*Dy1[4] + Q2[23]*Dy1[5];
}

void acado_multHxE( real_t* const Hx, real_t* const E, int row, int col )
{
acadoWorkspace.A[(row * 20 + 50) + (col * 2)] = + Hx[0]*E[0] + Hx[1]*E[2] + Hx[2]*E[4] + Hx[3]*E[6];
acadoWorkspace.A[(row * 20 + 50) + (col * 2 + 1)] = + Hx[0]*E[1] + Hx[1]*E[3] + Hx[2]*E[5] + Hx[3]*E[7];
acadoWorkspace.A[(row * 20 + 60) + (col * 2)] = + Hx[4]*E[0] + Hx[5]*E[2] + Hx[6]*E[4] + Hx[7]*E[6];
acadoWorkspace.A[(row * 20 + 60) + (col * 2 + 1)] = + Hx[4]*E[1] + Hx[5]*E[3] + Hx[6]*E[5] + Hx[7]*E[7];
}

void acado_macHxd( real_t* const Hx, real_t* const tmpd, real_t* const lbA, real_t* const ubA )
{
acadoWorkspace.evHxd[0] = + Hx[0]*tmpd[0] + Hx[1]*tmpd[1] + Hx[2]*tmpd[2] + Hx[3]*tmpd[3];
acadoWorkspace.evHxd[1] = + Hx[4]*tmpd[0] + Hx[5]*tmpd[1] + Hx[6]*tmpd[2] + Hx[7]*tmpd[3];
lbA[0] -= acadoWorkspace.evHxd[0];
lbA[1] -= acadoWorkspace.evHxd[1];
ubA[0] -= acadoWorkspace.evHxd[0];
ubA[1] -= acadoWorkspace.evHxd[1];
}

void acado_condensePrep(  )
{
int lRun1;
acado_moveGxT( acadoWorkspace.evGx, acadoWorkspace.C );
acado_multGxGx( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.C, &(acadoWorkspace.C[ 16 ]) );
acado_multGxGx( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.C[ 16 ]), &(acadoWorkspace.C[ 32 ]) );
acado_multGxGx( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.C[ 32 ]), &(acadoWorkspace.C[ 48 ]) );
acado_multGxGx( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.C[ 48 ]), &(acadoWorkspace.C[ 64 ]) );
/* Column: 0 */
acado_moveGuE( acadoWorkspace.evGu, acadoWorkspace.E );
acado_multGxGu( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.E, &(acadoWorkspace.E[ 8 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.E[ 8 ]), &(acadoWorkspace.E[ 16 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.E[ 16 ]), &(acadoWorkspace.E[ 24 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.E[ 24 ]), &(acadoWorkspace.E[ 32 ]) );

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 32 ]), acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.W1, 4, 0 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 24 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.W1, 3, 0 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 16 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 16 ]), acadoWorkspace.W1, 2, 0 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 8 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 8 ]), acadoWorkspace.W1, 1, 0 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 16 ]), acadoWorkspace.E, acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1_R1( acadoWorkspace.R1, acadoWorkspace.evGu, acadoWorkspace.W1, 0 );

/* Column: 1 */
acado_moveGuE( &(acadoWorkspace.evGu[ 8 ]), &(acadoWorkspace.E[ 40 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.E[ 40 ]), &(acadoWorkspace.E[ 48 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.E[ 48 ]), &(acadoWorkspace.E[ 56 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.E[ 56 ]), &(acadoWorkspace.E[ 64 ]) );

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 64 ]), acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.W1, 4, 1 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 56 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.W1, 3, 1 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 48 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 16 ]), acadoWorkspace.W1, 2, 1 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.E[ 40 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1_R1( &(acadoWorkspace.R1[ 4 ]), &(acadoWorkspace.evGu[ 8 ]), acadoWorkspace.W1, 1 );

/* Column: 2 */
acado_moveGuE( &(acadoWorkspace.evGu[ 16 ]), &(acadoWorkspace.E[ 72 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.E[ 72 ]), &(acadoWorkspace.E[ 80 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.E[ 80 ]), &(acadoWorkspace.E[ 88 ]) );

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 88 ]), acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.W1, 4, 2 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 80 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.W1, 3, 2 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.E[ 72 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1_R1( &(acadoWorkspace.R1[ 8 ]), &(acadoWorkspace.evGu[ 16 ]), acadoWorkspace.W1, 2 );

/* Column: 3 */
acado_moveGuE( &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.E[ 96 ]) );
acado_multGxGu( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.E[ 96 ]), &(acadoWorkspace.E[ 104 ]) );

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 104 ]), acadoWorkspace.W1 );
acado_multBTW1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.W1, 4, 3 );
acado_multGxTGu( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.W1, acadoWorkspace.W2 );
acado_multQEW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.E[ 96 ]), acadoWorkspace.W2, acadoWorkspace.W1 );
acado_multBTW1_R1( &(acadoWorkspace.R1[ 12 ]), &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.W1, 3 );

/* Column: 4 */
acado_moveGuE( &(acadoWorkspace.evGu[ 32 ]), &(acadoWorkspace.E[ 112 ]) );

acado_multGxGu( acadoWorkspace.QN1, &(acadoWorkspace.E[ 112 ]), acadoWorkspace.W1 );
acado_multBTW1_R1( &(acadoWorkspace.R1[ 16 ]), &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.W1, 4 );

acado_copyHTH( 0, 1 );
acado_copyHTH( 0, 2 );
acado_copyHTH( 1, 2 );
acado_copyHTH( 0, 3 );
acado_copyHTH( 1, 3 );
acado_copyHTH( 2, 3 );
acado_copyHTH( 0, 4 );
acado_copyHTH( 1, 4 );
acado_copyHTH( 2, 4 );
acado_copyHTH( 3, 4 );

acadoWorkspace.sbar[4] = acadoWorkspace.d[0];
acadoWorkspace.sbar[5] = acadoWorkspace.d[1];
acadoWorkspace.sbar[6] = acadoWorkspace.d[2];
acadoWorkspace.sbar[7] = acadoWorkspace.d[3];
acadoWorkspace.sbar[8] = acadoWorkspace.d[4];
acadoWorkspace.sbar[9] = acadoWorkspace.d[5];
acadoWorkspace.sbar[10] = acadoWorkspace.d[6];
acadoWorkspace.sbar[11] = acadoWorkspace.d[7];
acadoWorkspace.sbar[12] = acadoWorkspace.d[8];
acadoWorkspace.sbar[13] = acadoWorkspace.d[9];
acadoWorkspace.sbar[14] = acadoWorkspace.d[10];
acadoWorkspace.sbar[15] = acadoWorkspace.d[11];
acadoWorkspace.sbar[16] = acadoWorkspace.d[12];
acadoWorkspace.sbar[17] = acadoWorkspace.d[13];
acadoWorkspace.sbar[18] = acadoWorkspace.d[14];
acadoWorkspace.sbar[19] = acadoWorkspace.d[15];
acadoWorkspace.sbar[20] = acadoWorkspace.d[16];
acadoWorkspace.sbar[21] = acadoWorkspace.d[17];
acadoWorkspace.sbar[22] = acadoWorkspace.d[18];
acadoWorkspace.sbar[23] = acadoWorkspace.d[19];
acadoWorkspace.lb[0] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.lb[1] = (real_t)-1.0000000000000000e+01 - acadoVariables.u[1];
acadoWorkspace.lb[2] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.lb[3] = (real_t)-1.0000000000000000e+01 - acadoVariables.u[3];
acadoWorkspace.lb[4] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.lb[5] = (real_t)-1.0000000000000000e+01 - acadoVariables.u[5];
acadoWorkspace.lb[6] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.lb[7] = (real_t)-1.0000000000000000e+01 - acadoVariables.u[7];
acadoWorkspace.lb[8] = (real_t)-1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.lb[9] = (real_t)-1.0000000000000000e+01 - acadoVariables.u[9];
acadoWorkspace.ub[0] = (real_t)1.0000000000000000e+12 - acadoVariables.u[0];
acadoWorkspace.ub[1] = (real_t)1.0000000000000000e+01 - acadoVariables.u[1];
acadoWorkspace.ub[2] = (real_t)1.0000000000000000e+12 - acadoVariables.u[2];
acadoWorkspace.ub[3] = (real_t)1.0000000000000000e+01 - acadoVariables.u[3];
acadoWorkspace.ub[4] = (real_t)1.0000000000000000e+12 - acadoVariables.u[4];
acadoWorkspace.ub[5] = (real_t)1.0000000000000000e+01 - acadoVariables.u[5];
acadoWorkspace.ub[6] = (real_t)1.0000000000000000e+12 - acadoVariables.u[6];
acadoWorkspace.ub[7] = (real_t)1.0000000000000000e+01 - acadoVariables.u[7];
acadoWorkspace.ub[8] = (real_t)1.0000000000000000e+12 - acadoVariables.u[8];
acadoWorkspace.ub[9] = (real_t)1.0000000000000000e+01 - acadoVariables.u[9];

acadoWorkspace.A[0] = acadoWorkspace.E[6];
acadoWorkspace.A[1] = acadoWorkspace.E[7];

acadoWorkspace.A[10] = acadoWorkspace.E[14];
acadoWorkspace.A[11] = acadoWorkspace.E[15];
acadoWorkspace.A[12] = acadoWorkspace.E[46];
acadoWorkspace.A[13] = acadoWorkspace.E[47];

acadoWorkspace.A[20] = acadoWorkspace.E[22];
acadoWorkspace.A[21] = acadoWorkspace.E[23];
acadoWorkspace.A[22] = acadoWorkspace.E[54];
acadoWorkspace.A[23] = acadoWorkspace.E[55];
acadoWorkspace.A[24] = acadoWorkspace.E[78];
acadoWorkspace.A[25] = acadoWorkspace.E[79];

acadoWorkspace.A[30] = acadoWorkspace.E[30];
acadoWorkspace.A[31] = acadoWorkspace.E[31];
acadoWorkspace.A[32] = acadoWorkspace.E[62];
acadoWorkspace.A[33] = acadoWorkspace.E[63];
acadoWorkspace.A[34] = acadoWorkspace.E[86];
acadoWorkspace.A[35] = acadoWorkspace.E[87];
acadoWorkspace.A[36] = acadoWorkspace.E[102];
acadoWorkspace.A[37] = acadoWorkspace.E[103];

acadoWorkspace.A[40] = acadoWorkspace.E[38];
acadoWorkspace.A[41] = acadoWorkspace.E[39];
acadoWorkspace.A[42] = acadoWorkspace.E[70];
acadoWorkspace.A[43] = acadoWorkspace.E[71];
acadoWorkspace.A[44] = acadoWorkspace.E[94];
acadoWorkspace.A[45] = acadoWorkspace.E[95];
acadoWorkspace.A[46] = acadoWorkspace.E[110];
acadoWorkspace.A[47] = acadoWorkspace.E[111];
acadoWorkspace.A[48] = acadoWorkspace.E[118];
acadoWorkspace.A[49] = acadoWorkspace.E[119];


for (lRun1 = 0; lRun1 < 5; ++lRun1)
{
acadoWorkspace.conValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.conValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.conValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.conValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.conValueIn[4] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.conValueIn[5] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.conValueIn[6] = acadoVariables.od[lRun1 * 6];
acadoWorkspace.conValueIn[7] = acadoVariables.od[lRun1 * 6 + 1];
acadoWorkspace.conValueIn[8] = acadoVariables.od[lRun1 * 6 + 2];
acadoWorkspace.conValueIn[9] = acadoVariables.od[lRun1 * 6 + 3];
acadoWorkspace.conValueIn[10] = acadoVariables.od[lRun1 * 6 + 4];
acadoWorkspace.conValueIn[11] = acadoVariables.od[lRun1 * 6 + 5];
acado_evaluatePathConstraints( acadoWorkspace.conValueIn, acadoWorkspace.conValueOut );
acadoWorkspace.evH[lRun1 * 2] = acadoWorkspace.conValueOut[0];
acadoWorkspace.evH[lRun1 * 2 + 1] = acadoWorkspace.conValueOut[1];

acadoWorkspace.evHx[lRun1 * 8] = acadoWorkspace.conValueOut[2];
acadoWorkspace.evHx[lRun1 * 8 + 1] = acadoWorkspace.conValueOut[3];
acadoWorkspace.evHx[lRun1 * 8 + 2] = acadoWorkspace.conValueOut[4];
acadoWorkspace.evHx[lRun1 * 8 + 3] = acadoWorkspace.conValueOut[5];
acadoWorkspace.evHx[lRun1 * 8 + 4] = acadoWorkspace.conValueOut[6];
acadoWorkspace.evHx[lRun1 * 8 + 5] = acadoWorkspace.conValueOut[7];
acadoWorkspace.evHx[lRun1 * 8 + 6] = acadoWorkspace.conValueOut[8];
acadoWorkspace.evHx[lRun1 * 8 + 7] = acadoWorkspace.conValueOut[9];
acadoWorkspace.evHu[lRun1 * 4] = acadoWorkspace.conValueOut[10];
acadoWorkspace.evHu[lRun1 * 4 + 1] = acadoWorkspace.conValueOut[11];
acadoWorkspace.evHu[lRun1 * 4 + 2] = acadoWorkspace.conValueOut[12];
acadoWorkspace.evHu[lRun1 * 4 + 3] = acadoWorkspace.conValueOut[13];
}



acado_multHxE( &(acadoWorkspace.evHx[ 8 ]), acadoWorkspace.E, 1, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 8 ]), 2, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.E[ 40 ]), 2, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 16 ]), 3, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 48 ]), 3, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.E[ 72 ]), 3, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 24 ]), 4, 0 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 56 ]), 4, 1 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 80 ]), 4, 2 );
acado_multHxE( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.E[ 96 ]), 4, 3 );

acadoWorkspace.A[50] = acadoWorkspace.evHu[0];
acadoWorkspace.A[51] = acadoWorkspace.evHu[1];
acadoWorkspace.A[60] = acadoWorkspace.evHu[2];
acadoWorkspace.A[61] = acadoWorkspace.evHu[3];
acadoWorkspace.A[72] = acadoWorkspace.evHu[4];
acadoWorkspace.A[73] = acadoWorkspace.evHu[5];
acadoWorkspace.A[82] = acadoWorkspace.evHu[6];
acadoWorkspace.A[83] = acadoWorkspace.evHu[7];
acadoWorkspace.A[94] = acadoWorkspace.evHu[8];
acadoWorkspace.A[95] = acadoWorkspace.evHu[9];
acadoWorkspace.A[104] = acadoWorkspace.evHu[10];
acadoWorkspace.A[105] = acadoWorkspace.evHu[11];
acadoWorkspace.A[116] = acadoWorkspace.evHu[12];
acadoWorkspace.A[117] = acadoWorkspace.evHu[13];
acadoWorkspace.A[126] = acadoWorkspace.evHu[14];
acadoWorkspace.A[127] = acadoWorkspace.evHu[15];
acadoWorkspace.A[138] = acadoWorkspace.evHu[16];
acadoWorkspace.A[139] = acadoWorkspace.evHu[17];
acadoWorkspace.A[148] = acadoWorkspace.evHu[18];
acadoWorkspace.A[149] = acadoWorkspace.evHu[19];
acadoWorkspace.lbA[5] = - acadoWorkspace.evH[0];
acadoWorkspace.lbA[6] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[1];
acadoWorkspace.lbA[7] = - acadoWorkspace.evH[2];
acadoWorkspace.lbA[8] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[3];
acadoWorkspace.lbA[9] = - acadoWorkspace.evH[4];
acadoWorkspace.lbA[10] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[5];
acadoWorkspace.lbA[11] = - acadoWorkspace.evH[6];
acadoWorkspace.lbA[12] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[7];
acadoWorkspace.lbA[13] = - acadoWorkspace.evH[8];
acadoWorkspace.lbA[14] = (real_t)-1.0000000000000000e+12 - acadoWorkspace.evH[9];

acadoWorkspace.ubA[5] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[0];
acadoWorkspace.ubA[6] = - acadoWorkspace.evH[1];
acadoWorkspace.ubA[7] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[2];
acadoWorkspace.ubA[8] = - acadoWorkspace.evH[3];
acadoWorkspace.ubA[9] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[4];
acadoWorkspace.ubA[10] = - acadoWorkspace.evH[5];
acadoWorkspace.ubA[11] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[6];
acadoWorkspace.ubA[12] = - acadoWorkspace.evH[7];
acadoWorkspace.ubA[13] = (real_t)1.0000000000000000e+12 - acadoWorkspace.evH[8];
acadoWorkspace.ubA[14] = - acadoWorkspace.evH[9];

}

void acado_condenseFdb(  )
{
real_t tmp;

acadoWorkspace.Dx0[0] = acadoVariables.x0[0] - acadoVariables.x[0];
acadoWorkspace.Dx0[1] = acadoVariables.x0[1] - acadoVariables.x[1];
acadoWorkspace.Dx0[2] = acadoVariables.x0[2] - acadoVariables.x[2];
acadoWorkspace.Dx0[3] = acadoVariables.x0[3] - acadoVariables.x[3];
acadoWorkspace.Dy[0] -= acadoVariables.y[0];
acadoWorkspace.Dy[1] -= acadoVariables.y[1];
acadoWorkspace.Dy[2] -= acadoVariables.y[2];
acadoWorkspace.Dy[3] -= acadoVariables.y[3];
acadoWorkspace.Dy[4] -= acadoVariables.y[4];
acadoWorkspace.Dy[5] -= acadoVariables.y[5];
acadoWorkspace.Dy[6] -= acadoVariables.y[6];
acadoWorkspace.Dy[7] -= acadoVariables.y[7];
acadoWorkspace.Dy[8] -= acadoVariables.y[8];
acadoWorkspace.Dy[9] -= acadoVariables.y[9];
acadoWorkspace.Dy[10] -= acadoVariables.y[10];
acadoWorkspace.Dy[11] -= acadoVariables.y[11];
acadoWorkspace.Dy[12] -= acadoVariables.y[12];
acadoWorkspace.Dy[13] -= acadoVariables.y[13];
acadoWorkspace.Dy[14] -= acadoVariables.y[14];
acadoWorkspace.Dy[15] -= acadoVariables.y[15];
acadoWorkspace.Dy[16] -= acadoVariables.y[16];
acadoWorkspace.Dy[17] -= acadoVariables.y[17];
acadoWorkspace.Dy[18] -= acadoVariables.y[18];
acadoWorkspace.Dy[19] -= acadoVariables.y[19];
acadoWorkspace.Dy[20] -= acadoVariables.y[20];
acadoWorkspace.Dy[21] -= acadoVariables.y[21];
acadoWorkspace.Dy[22] -= acadoVariables.y[22];
acadoWorkspace.Dy[23] -= acadoVariables.y[23];
acadoWorkspace.Dy[24] -= acadoVariables.y[24];
acadoWorkspace.Dy[25] -= acadoVariables.y[25];
acadoWorkspace.Dy[26] -= acadoVariables.y[26];
acadoWorkspace.Dy[27] -= acadoVariables.y[27];
acadoWorkspace.Dy[28] -= acadoVariables.y[28];
acadoWorkspace.Dy[29] -= acadoVariables.y[29];
acadoWorkspace.DyN[0] -= acadoVariables.yN[0];
acadoWorkspace.DyN[1] -= acadoVariables.yN[1];
acadoWorkspace.DyN[2] -= acadoVariables.yN[2];
acadoWorkspace.DyN[3] -= acadoVariables.yN[3];

acado_multRDy( acadoWorkspace.R2, acadoWorkspace.Dy, acadoWorkspace.g );
acado_multRDy( &(acadoWorkspace.R2[ 12 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.g[ 2 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 24 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.g[ 4 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 36 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.g[ 6 ]) );
acado_multRDy( &(acadoWorkspace.R2[ 48 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.g[ 8 ]) );

acado_multQDy( acadoWorkspace.Q2, acadoWorkspace.Dy, acadoWorkspace.QDy );
acado_multQDy( &(acadoWorkspace.Q2[ 24 ]), &(acadoWorkspace.Dy[ 6 ]), &(acadoWorkspace.QDy[ 4 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 48 ]), &(acadoWorkspace.Dy[ 12 ]), &(acadoWorkspace.QDy[ 8 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 72 ]), &(acadoWorkspace.Dy[ 18 ]), &(acadoWorkspace.QDy[ 12 ]) );
acado_multQDy( &(acadoWorkspace.Q2[ 96 ]), &(acadoWorkspace.Dy[ 24 ]), &(acadoWorkspace.QDy[ 16 ]) );

acadoWorkspace.QDy[20] = + acadoWorkspace.QN2[0]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[1]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[2]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[3]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[21] = + acadoWorkspace.QN2[4]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[5]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[6]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[7]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[22] = + acadoWorkspace.QN2[8]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[9]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[10]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[11]*acadoWorkspace.DyN[3];
acadoWorkspace.QDy[23] = + acadoWorkspace.QN2[12]*acadoWorkspace.DyN[0] + acadoWorkspace.QN2[13]*acadoWorkspace.DyN[1] + acadoWorkspace.QN2[14]*acadoWorkspace.DyN[2] + acadoWorkspace.QN2[15]*acadoWorkspace.DyN[3];

acadoWorkspace.sbar[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.sbar[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.sbar[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.sbar[3] = acadoWorkspace.Dx0[3];
acado_macASbar( acadoWorkspace.evGx, acadoWorkspace.sbar, &(acadoWorkspace.sbar[ 4 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.sbar[ 4 ]), &(acadoWorkspace.sbar[ 8 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.sbar[ 8 ]), &(acadoWorkspace.sbar[ 12 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.sbar[ 12 ]), &(acadoWorkspace.sbar[ 16 ]) );
acado_macASbar( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.sbar[ 16 ]), &(acadoWorkspace.sbar[ 20 ]) );

acadoWorkspace.w1[0] = + acadoWorkspace.QN1[0]*acadoWorkspace.sbar[20] + acadoWorkspace.QN1[1]*acadoWorkspace.sbar[21] + acadoWorkspace.QN1[2]*acadoWorkspace.sbar[22] + acadoWorkspace.QN1[3]*acadoWorkspace.sbar[23] + acadoWorkspace.QDy[20];
acadoWorkspace.w1[1] = + acadoWorkspace.QN1[4]*acadoWorkspace.sbar[20] + acadoWorkspace.QN1[5]*acadoWorkspace.sbar[21] + acadoWorkspace.QN1[6]*acadoWorkspace.sbar[22] + acadoWorkspace.QN1[7]*acadoWorkspace.sbar[23] + acadoWorkspace.QDy[21];
acadoWorkspace.w1[2] = + acadoWorkspace.QN1[8]*acadoWorkspace.sbar[20] + acadoWorkspace.QN1[9]*acadoWorkspace.sbar[21] + acadoWorkspace.QN1[10]*acadoWorkspace.sbar[22] + acadoWorkspace.QN1[11]*acadoWorkspace.sbar[23] + acadoWorkspace.QDy[22];
acadoWorkspace.w1[3] = + acadoWorkspace.QN1[12]*acadoWorkspace.sbar[20] + acadoWorkspace.QN1[13]*acadoWorkspace.sbar[21] + acadoWorkspace.QN1[14]*acadoWorkspace.sbar[22] + acadoWorkspace.QN1[15]*acadoWorkspace.sbar[23] + acadoWorkspace.QDy[23];
acado_macBTw1( &(acadoWorkspace.evGu[ 32 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 8 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 64 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 16 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 64 ]), &(acadoWorkspace.sbar[ 16 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 24 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 6 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 48 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 12 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 48 ]), &(acadoWorkspace.sbar[ 12 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 16 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 4 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 32 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 8 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 32 ]), &(acadoWorkspace.sbar[ 8 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( &(acadoWorkspace.evGu[ 8 ]), acadoWorkspace.w1, &(acadoWorkspace.g[ 2 ]) );
acado_macATw1QDy( &(acadoWorkspace.evGx[ 16 ]), acadoWorkspace.w1, &(acadoWorkspace.QDy[ 4 ]), acadoWorkspace.w2 );
acado_macQSbarW2( &(acadoWorkspace.Q1[ 16 ]), &(acadoWorkspace.sbar[ 4 ]), acadoWorkspace.w2, acadoWorkspace.w1 );
acado_macBTw1( acadoWorkspace.evGu, acadoWorkspace.w1, acadoWorkspace.g );


tmp = acadoWorkspace.sbar[7] + acadoVariables.x[7];
acadoWorkspace.lbA[0] = - tmp;
acadoWorkspace.ubA[0] = (real_t)1.0000000000000000e+02 - tmp;
tmp = acadoWorkspace.sbar[11] + acadoVariables.x[11];
acadoWorkspace.lbA[1] = - tmp;
acadoWorkspace.ubA[1] = (real_t)1.0000000000000000e+02 - tmp;
tmp = acadoWorkspace.sbar[15] + acadoVariables.x[15];
acadoWorkspace.lbA[2] = - tmp;
acadoWorkspace.ubA[2] = (real_t)1.0000000000000000e+02 - tmp;
tmp = acadoWorkspace.sbar[19] + acadoVariables.x[19];
acadoWorkspace.lbA[3] = - tmp;
acadoWorkspace.ubA[3] = (real_t)1.0000000000000000e+02 - tmp;
tmp = acadoWorkspace.sbar[23] + acadoVariables.x[23];
acadoWorkspace.lbA[4] = - tmp;
acadoWorkspace.ubA[4] = (real_t)1.0000000000000000e+02 - tmp;

acado_macHxd( acadoWorkspace.evHx, acadoWorkspace.sbar, &(acadoWorkspace.lbA[ 5 ]), &(acadoWorkspace.ubA[ 5 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 8 ]), &(acadoWorkspace.sbar[ 4 ]), &(acadoWorkspace.lbA[ 7 ]), &(acadoWorkspace.ubA[ 7 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 16 ]), &(acadoWorkspace.sbar[ 8 ]), &(acadoWorkspace.lbA[ 9 ]), &(acadoWorkspace.ubA[ 9 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 24 ]), &(acadoWorkspace.sbar[ 12 ]), &(acadoWorkspace.lbA[ 11 ]), &(acadoWorkspace.ubA[ 11 ]) );
acado_macHxd( &(acadoWorkspace.evHx[ 32 ]), &(acadoWorkspace.sbar[ 16 ]), &(acadoWorkspace.lbA[ 13 ]), &(acadoWorkspace.ubA[ 13 ]) );

}

void acado_expand(  )
{
acadoVariables.u[0] += acadoWorkspace.x[0];
acadoVariables.u[1] += acadoWorkspace.x[1];
acadoVariables.u[2] += acadoWorkspace.x[2];
acadoVariables.u[3] += acadoWorkspace.x[3];
acadoVariables.u[4] += acadoWorkspace.x[4];
acadoVariables.u[5] += acadoWorkspace.x[5];
acadoVariables.u[6] += acadoWorkspace.x[6];
acadoVariables.u[7] += acadoWorkspace.x[7];
acadoVariables.u[8] += acadoWorkspace.x[8];
acadoVariables.u[9] += acadoWorkspace.x[9];
acadoWorkspace.sbar[0] = acadoWorkspace.Dx0[0];
acadoWorkspace.sbar[1] = acadoWorkspace.Dx0[1];
acadoWorkspace.sbar[2] = acadoWorkspace.Dx0[2];
acadoWorkspace.sbar[3] = acadoWorkspace.Dx0[3];
acadoWorkspace.sbar[4] = acadoWorkspace.d[0];
acadoWorkspace.sbar[5] = acadoWorkspace.d[1];
acadoWorkspace.sbar[6] = acadoWorkspace.d[2];
acadoWorkspace.sbar[7] = acadoWorkspace.d[3];
acadoWorkspace.sbar[8] = acadoWorkspace.d[4];
acadoWorkspace.sbar[9] = acadoWorkspace.d[5];
acadoWorkspace.sbar[10] = acadoWorkspace.d[6];
acadoWorkspace.sbar[11] = acadoWorkspace.d[7];
acadoWorkspace.sbar[12] = acadoWorkspace.d[8];
acadoWorkspace.sbar[13] = acadoWorkspace.d[9];
acadoWorkspace.sbar[14] = acadoWorkspace.d[10];
acadoWorkspace.sbar[15] = acadoWorkspace.d[11];
acadoWorkspace.sbar[16] = acadoWorkspace.d[12];
acadoWorkspace.sbar[17] = acadoWorkspace.d[13];
acadoWorkspace.sbar[18] = acadoWorkspace.d[14];
acadoWorkspace.sbar[19] = acadoWorkspace.d[15];
acadoWorkspace.sbar[20] = acadoWorkspace.d[16];
acadoWorkspace.sbar[21] = acadoWorkspace.d[17];
acadoWorkspace.sbar[22] = acadoWorkspace.d[18];
acadoWorkspace.sbar[23] = acadoWorkspace.d[19];
acado_expansionStep( acadoWorkspace.evGx, acadoWorkspace.evGu, acadoWorkspace.x, acadoWorkspace.sbar, &(acadoWorkspace.sbar[ 4 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 16 ]), &(acadoWorkspace.evGu[ 8 ]), &(acadoWorkspace.x[ 2 ]), &(acadoWorkspace.sbar[ 4 ]), &(acadoWorkspace.sbar[ 8 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 32 ]), &(acadoWorkspace.evGu[ 16 ]), &(acadoWorkspace.x[ 4 ]), &(acadoWorkspace.sbar[ 8 ]), &(acadoWorkspace.sbar[ 12 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 48 ]), &(acadoWorkspace.evGu[ 24 ]), &(acadoWorkspace.x[ 6 ]), &(acadoWorkspace.sbar[ 12 ]), &(acadoWorkspace.sbar[ 16 ]) );
acado_expansionStep( &(acadoWorkspace.evGx[ 64 ]), &(acadoWorkspace.evGu[ 32 ]), &(acadoWorkspace.x[ 8 ]), &(acadoWorkspace.sbar[ 16 ]), &(acadoWorkspace.sbar[ 20 ]) );
acadoVariables.x[0] += acadoWorkspace.sbar[0];
acadoVariables.x[1] += acadoWorkspace.sbar[1];
acadoVariables.x[2] += acadoWorkspace.sbar[2];
acadoVariables.x[3] += acadoWorkspace.sbar[3];
acadoVariables.x[4] += acadoWorkspace.sbar[4];
acadoVariables.x[5] += acadoWorkspace.sbar[5];
acadoVariables.x[6] += acadoWorkspace.sbar[6];
acadoVariables.x[7] += acadoWorkspace.sbar[7];
acadoVariables.x[8] += acadoWorkspace.sbar[8];
acadoVariables.x[9] += acadoWorkspace.sbar[9];
acadoVariables.x[10] += acadoWorkspace.sbar[10];
acadoVariables.x[11] += acadoWorkspace.sbar[11];
acadoVariables.x[12] += acadoWorkspace.sbar[12];
acadoVariables.x[13] += acadoWorkspace.sbar[13];
acadoVariables.x[14] += acadoWorkspace.sbar[14];
acadoVariables.x[15] += acadoWorkspace.sbar[15];
acadoVariables.x[16] += acadoWorkspace.sbar[16];
acadoVariables.x[17] += acadoWorkspace.sbar[17];
acadoVariables.x[18] += acadoWorkspace.sbar[18];
acadoVariables.x[19] += acadoWorkspace.sbar[19];
acadoVariables.x[20] += acadoWorkspace.sbar[20];
acadoVariables.x[21] += acadoWorkspace.sbar[21];
acadoVariables.x[22] += acadoWorkspace.sbar[22];
acadoVariables.x[23] += acadoWorkspace.sbar[23];
}

int acado_preparationStep(  )
{
int ret;

ret = acado_modelSimulation();
acado_evaluateObjective(  );
acado_condensePrep(  );
return ret;
}

int acado_feedbackStep(  )
{
int tmp;

acado_condenseFdb(  );

tmp = acado_solve( );

acado_expand(  );
return tmp;
}

int acado_initializeSolver(  )
{
int ret;

/* This is a function which must be called once before any other function call! */


ret = 0;

memset(&acadoWorkspace, 0, sizeof( acadoWorkspace ));
return ret;
}

void acado_initializeNodesByForwardSimulation(  )
{
int index;
for (index = 0; index < 5; ++index)
{
acadoWorkspace.state[0] = acadoVariables.x[index * 4];
acadoWorkspace.state[1] = acadoVariables.x[index * 4 + 1];
acadoWorkspace.state[2] = acadoVariables.x[index * 4 + 2];
acadoWorkspace.state[3] = acadoVariables.x[index * 4 + 3];
acadoWorkspace.state[28] = acadoVariables.u[index * 2];
acadoWorkspace.state[29] = acadoVariables.u[index * 2 + 1];
acadoWorkspace.state[30] = acadoVariables.od[index * 6];
acadoWorkspace.state[31] = acadoVariables.od[index * 6 + 1];
acadoWorkspace.state[32] = acadoVariables.od[index * 6 + 2];
acadoWorkspace.state[33] = acadoVariables.od[index * 6 + 3];
acadoWorkspace.state[34] = acadoVariables.od[index * 6 + 4];
acadoWorkspace.state[35] = acadoVariables.od[index * 6 + 5];

acado_integrate(acadoWorkspace.state, index == 0);

acadoVariables.x[index * 4 + 4] = acadoWorkspace.state[0];
acadoVariables.x[index * 4 + 5] = acadoWorkspace.state[1];
acadoVariables.x[index * 4 + 6] = acadoWorkspace.state[2];
acadoVariables.x[index * 4 + 7] = acadoWorkspace.state[3];
}
}

void acado_shiftStates( int strategy, real_t* const xEnd, real_t* const uEnd )
{
int index;
for (index = 0; index < 5; ++index)
{
acadoVariables.x[index * 4] = acadoVariables.x[index * 4 + 4];
acadoVariables.x[index * 4 + 1] = acadoVariables.x[index * 4 + 5];
acadoVariables.x[index * 4 + 2] = acadoVariables.x[index * 4 + 6];
acadoVariables.x[index * 4 + 3] = acadoVariables.x[index * 4 + 7];
}

if (strategy == 1 && xEnd != 0)
{
acadoVariables.x[20] = xEnd[0];
acadoVariables.x[21] = xEnd[1];
acadoVariables.x[22] = xEnd[2];
acadoVariables.x[23] = xEnd[3];
}
else if (strategy == 2) 
{
acadoWorkspace.state[0] = acadoVariables.x[20];
acadoWorkspace.state[1] = acadoVariables.x[21];
acadoWorkspace.state[2] = acadoVariables.x[22];
acadoWorkspace.state[3] = acadoVariables.x[23];
if (uEnd != 0)
{
acadoWorkspace.state[28] = uEnd[0];
acadoWorkspace.state[29] = uEnd[1];
}
else
{
acadoWorkspace.state[28] = acadoVariables.u[8];
acadoWorkspace.state[29] = acadoVariables.u[9];
}
acadoWorkspace.state[30] = acadoVariables.od[30];
acadoWorkspace.state[31] = acadoVariables.od[31];
acadoWorkspace.state[32] = acadoVariables.od[32];
acadoWorkspace.state[33] = acadoVariables.od[33];
acadoWorkspace.state[34] = acadoVariables.od[34];
acadoWorkspace.state[35] = acadoVariables.od[35];

acado_integrate(acadoWorkspace.state, 1);

acadoVariables.x[20] = acadoWorkspace.state[0];
acadoVariables.x[21] = acadoWorkspace.state[1];
acadoVariables.x[22] = acadoWorkspace.state[2];
acadoVariables.x[23] = acadoWorkspace.state[3];
}
}

void acado_shiftControls( real_t* const uEnd )
{
int index;
for (index = 0; index < 4; ++index)
{
acadoVariables.u[index * 2] = acadoVariables.u[index * 2 + 2];
acadoVariables.u[index * 2 + 1] = acadoVariables.u[index * 2 + 3];
}

if (uEnd != 0)
{
acadoVariables.u[8] = uEnd[0];
acadoVariables.u[9] = uEnd[1];
}
}

real_t acado_getKKT(  )
{
real_t kkt;

int index;
real_t prd;

kkt = + acadoWorkspace.g[0]*acadoWorkspace.x[0] + acadoWorkspace.g[1]*acadoWorkspace.x[1] + acadoWorkspace.g[2]*acadoWorkspace.x[2] + acadoWorkspace.g[3]*acadoWorkspace.x[3] + acadoWorkspace.g[4]*acadoWorkspace.x[4] + acadoWorkspace.g[5]*acadoWorkspace.x[5] + acadoWorkspace.g[6]*acadoWorkspace.x[6] + acadoWorkspace.g[7]*acadoWorkspace.x[7] + acadoWorkspace.g[8]*acadoWorkspace.x[8] + acadoWorkspace.g[9]*acadoWorkspace.x[9];
kkt = fabs( kkt );
for (index = 0; index < 10; ++index)
{
prd = acadoWorkspace.y[index];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lb[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ub[index] * prd);
}
for (index = 0; index < 15; ++index)
{
prd = acadoWorkspace.y[index + 10];
if (prd > 1e-12)
kkt += fabs(acadoWorkspace.lbA[index] * prd);
else if (prd < -1e-12)
kkt += fabs(acadoWorkspace.ubA[index] * prd);
}
return kkt;
}

real_t acado_getObjective(  )
{
real_t objVal;

int lRun1;
/** Row vector of size: 6 */
real_t tmpDy[ 6 ];

/** Row vector of size: 4 */
real_t tmpDyN[ 4 ];

for (lRun1 = 0; lRun1 < 5; ++lRun1)
{
acadoWorkspace.objValueIn[0] = acadoVariables.x[lRun1 * 4];
acadoWorkspace.objValueIn[1] = acadoVariables.x[lRun1 * 4 + 1];
acadoWorkspace.objValueIn[2] = acadoVariables.x[lRun1 * 4 + 2];
acadoWorkspace.objValueIn[3] = acadoVariables.x[lRun1 * 4 + 3];
acadoWorkspace.objValueIn[4] = acadoVariables.u[lRun1 * 2];
acadoWorkspace.objValueIn[5] = acadoVariables.u[lRun1 * 2 + 1];
acadoWorkspace.objValueIn[6] = acadoVariables.od[lRun1 * 6];
acadoWorkspace.objValueIn[7] = acadoVariables.od[lRun1 * 6 + 1];
acadoWorkspace.objValueIn[8] = acadoVariables.od[lRun1 * 6 + 2];
acadoWorkspace.objValueIn[9] = acadoVariables.od[lRun1 * 6 + 3];
acadoWorkspace.objValueIn[10] = acadoVariables.od[lRun1 * 6 + 4];
acadoWorkspace.objValueIn[11] = acadoVariables.od[lRun1 * 6 + 5];

acado_evaluateLSQ( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.Dy[lRun1 * 6] = acadoWorkspace.objValueOut[0] - acadoVariables.y[lRun1 * 6];
acadoWorkspace.Dy[lRun1 * 6 + 1] = acadoWorkspace.objValueOut[1] - acadoVariables.y[lRun1 * 6 + 1];
acadoWorkspace.Dy[lRun1 * 6 + 2] = acadoWorkspace.objValueOut[2] - acadoVariables.y[lRun1 * 6 + 2];
acadoWorkspace.Dy[lRun1 * 6 + 3] = acadoWorkspace.objValueOut[3] - acadoVariables.y[lRun1 * 6 + 3];
acadoWorkspace.Dy[lRun1 * 6 + 4] = acadoWorkspace.objValueOut[4] - acadoVariables.y[lRun1 * 6 + 4];
acadoWorkspace.Dy[lRun1 * 6 + 5] = acadoWorkspace.objValueOut[5] - acadoVariables.y[lRun1 * 6 + 5];
}
acadoWorkspace.objValueIn[0] = acadoVariables.x[20];
acadoWorkspace.objValueIn[1] = acadoVariables.x[21];
acadoWorkspace.objValueIn[2] = acadoVariables.x[22];
acadoWorkspace.objValueIn[3] = acadoVariables.x[23];
acadoWorkspace.objValueIn[4] = acadoVariables.od[30];
acadoWorkspace.objValueIn[5] = acadoVariables.od[31];
acadoWorkspace.objValueIn[6] = acadoVariables.od[32];
acadoWorkspace.objValueIn[7] = acadoVariables.od[33];
acadoWorkspace.objValueIn[8] = acadoVariables.od[34];
acadoWorkspace.objValueIn[9] = acadoVariables.od[35];
acado_evaluateLSQEndTerm( acadoWorkspace.objValueIn, acadoWorkspace.objValueOut );
acadoWorkspace.DyN[0] = acadoWorkspace.objValueOut[0] - acadoVariables.yN[0];
acadoWorkspace.DyN[1] = acadoWorkspace.objValueOut[1] - acadoVariables.yN[1];
acadoWorkspace.DyN[2] = acadoWorkspace.objValueOut[2] - acadoVariables.yN[2];
acadoWorkspace.DyN[3] = acadoWorkspace.objValueOut[3] - acadoVariables.yN[3];
objVal = 0.0000000000000000e+00;
for (lRun1 = 0; lRun1 < 5; ++lRun1)
{
tmpDy[0] = + acadoWorkspace.Dy[lRun1 * 6]*acadoVariables.W[0];
tmpDy[1] = + acadoWorkspace.Dy[lRun1 * 6 + 1]*acadoVariables.W[7];
tmpDy[2] = + acadoWorkspace.Dy[lRun1 * 6 + 2]*acadoVariables.W[14];
tmpDy[3] = + acadoWorkspace.Dy[lRun1 * 6 + 3]*acadoVariables.W[21];
tmpDy[4] = + acadoWorkspace.Dy[lRun1 * 6 + 4]*acadoVariables.W[28];
tmpDy[5] = + acadoWorkspace.Dy[lRun1 * 6 + 5]*acadoVariables.W[35];
objVal += + acadoWorkspace.Dy[lRun1 * 6]*tmpDy[0] + acadoWorkspace.Dy[lRun1 * 6 + 1]*tmpDy[1] + acadoWorkspace.Dy[lRun1 * 6 + 2]*tmpDy[2] + acadoWorkspace.Dy[lRun1 * 6 + 3]*tmpDy[3] + acadoWorkspace.Dy[lRun1 * 6 + 4]*tmpDy[4] + acadoWorkspace.Dy[lRun1 * 6 + 5]*tmpDy[5];
}

tmpDyN[0] = + acadoWorkspace.DyN[0]*acadoVariables.WN[0];
tmpDyN[1] = + acadoWorkspace.DyN[1]*acadoVariables.WN[5];
tmpDyN[2] = + acadoWorkspace.DyN[2]*acadoVariables.WN[10];
tmpDyN[3] = + acadoWorkspace.DyN[3]*acadoVariables.WN[15];
objVal += + acadoWorkspace.DyN[0]*tmpDyN[0] + acadoWorkspace.DyN[1]*tmpDyN[1] + acadoWorkspace.DyN[2]*tmpDyN[2] + acadoWorkspace.DyN[3]*tmpDyN[3];

objVal *= 0.5;
return objVal;
}

