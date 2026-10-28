#include "textflag.h"

// Quantization of the attention operands in AVX2 (see attention8.go).
// VCVTPS2DQ rounds to nearest even, as roundHalfEven; VPACKSSDW then
// VPACKSSWB narrow 32 int32 to int8, in an order VPERMD restores.

DATA quantPerm<>+0(SB)/4, $0
DATA quantPerm<>+4(SB)/4, $4
DATA quantPerm<>+8(SB)/4, $1
DATA quantPerm<>+12(SB)/4, $5
DATA quantPerm<>+16(SB)/4, $2
DATA quantPerm<>+20(SB)/4, $6
DATA quantPerm<>+24(SB)/4, $3
DATA quantPerm<>+28(SB)/4, $7
GLOBL quantPerm<>(SB), RODATA|NOPTR, $32

// func quantRowAVX2(dst *int8, x, mean *float32, n int, flip int) (scale float32, sum int32)
//
// With c = x − mean (n values, a multiple of 32): scale = max|c|/127 and
// dst[i] = round(c[i]/scale), its top bit flipped if flip (uint8 with the
// 128 offset); sum is the sum of the int8 values. If c is zero, scale is
// 0 and dst is not written.
TEXT ·quantRowAVX2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ x+8(FP), SI
	MOVQ mean+16(FP), DX
	MOVQ n+24(FP), CX
	MOVL $0x7fffffff, AX
	VMOVD AX, X15
	VPBROADCASTD X15, Y15
	VXORPS Y0, Y0, Y0
	XORQ BX, BX

amax:
	VMOVUPS (SI)(BX*4), Y1
	VSUBPS  (DX)(BX*4), Y1, Y1
	VANDPS  Y15, Y1, Y1
	VMAXPS  Y1, Y0, Y0
	ADDQ    $8, BX
	CMPQ    BX, CX
	JLT     amax

	VEXTRACTF128 $1, Y0, X1
	VMAXPS       X1, X0, X0
	VPERMILPS    $0x4E, X0, X1
	VMAXPS       X1, X0, X0
	VPERMILPS    $0xB1, X0, X1
	VMAXPS       X1, X0, X0
	VXORPS       X2, X2, X2
	VUCOMISS     X2, X0
	JEQ          zero

	MOVL   $0x42FE0000, AX // 127
	VMOVD  AX, X3
	VDIVSS X3, X0, X4      // scale = max/127
	VMOVSS X4, scale+40(FP)
	MOVL   $0x3F800000, AX // 1
	VMOVD  AX, X5
	VDIVSS X4, X5, X6      // 1/scale
	VBROADCASTSS X6, Y6

	VPXOR Y14, Y14, Y14
	MOVQ  flip+32(FP), AX
	TESTQ AX, AX
	JEQ   noflip
	MOVL  $0x80808080, AX
	VMOVD AX, X14
	VPBROADCASTD X14, Y14

noflip:
	VMOVDQU quantPerm<>(SB), Y12
	VPXOR   Y13, Y13, Y13
	XORQ    BX, BX

qloop:
	VMOVUPS   (SI)(BX*4), Y0
	VMOVUPS   32(SI)(BX*4), Y1
	VMOVUPS   64(SI)(BX*4), Y2
	VMOVUPS   96(SI)(BX*4), Y3
	VSUBPS    (DX)(BX*4), Y0, Y0
	VSUBPS    32(DX)(BX*4), Y1, Y1
	VSUBPS    64(DX)(BX*4), Y2, Y2
	VSUBPS    96(DX)(BX*4), Y3, Y3
	VMULPS    Y6, Y0, Y0
	VMULPS    Y6, Y1, Y1
	VMULPS    Y6, Y2, Y2
	VMULPS    Y6, Y3, Y3
	VCVTPS2DQ Y0, Y0
	VCVTPS2DQ Y1, Y1
	VCVTPS2DQ Y2, Y2
	VCVTPS2DQ Y3, Y3
	VPADDD    Y0, Y13, Y13
	VPADDD    Y1, Y13, Y13
	VPADDD    Y2, Y13, Y13
	VPADDD    Y3, Y13, Y13
	VPACKSSDW Y1, Y0, Y0
	VPACKSSDW Y3, Y2, Y2
	VPACKSSWB Y2, Y0, Y0
	VPERMD    Y0, Y12, Y0
	VPXOR     Y14, Y0, Y0
	VMOVDQU   Y0, (DI)(BX*1)
	ADDQ      $32, BX
	CMPQ      BX, CX
	JLT       qloop

	VEXTRACTI128 $1, Y13, X1
	VPADDD       X1, X13, X13
	VPSHUFD      $0x4E, X13, X1
	VPADDD       X1, X13, X13
	VPSHUFD      $0xB1, X13, X1
	VPADDD       X1, X13, X13
	VMOVD        X13, AX
	MOVL         AX, sum+44(FP)
	VZEROUPPER
	RET

zero:
	MOVL $0, scale+40(FP)
	MOVL $0, sum+44(FP)
	VZEROUPPER
	RET

// func quantColsAVX2(dst *int8, x, inv *float32, n int)
//
// dst[i] = round(x[i]·inv[i]), n a multiple of 32, |x[i]·inv[i]| <= 127.
TEXT ·quantColsAVX2(SB), NOSPLIT, $0-32
	MOVQ    dst+0(FP), DI
	MOVQ    x+8(FP), SI
	MOVQ    inv+16(FP), DX
	MOVQ    n+24(FP), CX
	VMOVDQU quantPerm<>(SB), Y12
	XORQ    BX, BX

cloop:
	VMOVUPS   (SI)(BX*4), Y0
	VMOVUPS   32(SI)(BX*4), Y1
	VMOVUPS   64(SI)(BX*4), Y2
	VMOVUPS   96(SI)(BX*4), Y3
	VMULPS    (DX)(BX*4), Y0, Y0
	VMULPS    32(DX)(BX*4), Y1, Y1
	VMULPS    64(DX)(BX*4), Y2, Y2
	VMULPS    96(DX)(BX*4), Y3, Y3
	VCVTPS2DQ Y0, Y0
	VCVTPS2DQ Y1, Y1
	VCVTPS2DQ Y2, Y2
	VCVTPS2DQ Y3, Y3
	VPACKSSDW Y1, Y0, Y0
	VPACKSSDW Y3, Y2, Y2
	VPACKSSWB Y2, Y0, Y0
	VPERMD    Y0, Y12, Y0
	VMOVDQU   Y0, (DI)(BX*1)
	ADDQ      $32, BX
	CMPQ      BX, CX
	JLT       cloop
	VZEROUPPER
	RET

// func interleave4AVX2(dst, src *int8, n, stride int)
//
// src holds 4 rows of n bytes (n a multiple of 16), r_t at src + t·n.
// For each chunk c of 16 columns, dst + c·stride receives their 64 bytes
// column by column: r0[j] r1[j] r2[j] r3[j] for j in the chunk.
TEXT ·interleave4AVX2(SB), NOSPLIT, $0-32
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ n+16(FP), CX
	MOVQ stride+24(FP), DX
	LEAQ (SI)(CX*1), R8
	LEAQ (R8)(CX*1), R9
	LEAQ (R9)(CX*1), R10
	XORQ BX, BX

iloop:
	VMOVDQU    (SI)(BX*1), X0
	VMOVDQU    (R8)(BX*1), X1
	VMOVDQU    (R9)(BX*1), X2
	VMOVDQU    (R10)(BX*1), X3
	VPUNPCKLBW X1, X0, X4
	VPUNPCKHBW X1, X0, X5
	VPUNPCKLBW X3, X2, X6
	VPUNPCKHBW X3, X2, X7
	VPUNPCKLWD X6, X4, X8
	VPUNPCKHWD X6, X4, X9
	VPUNPCKLWD X7, X5, X10
	VPUNPCKHWD X7, X5, X11
	VMOVDQU    X8, (DI)
	VMOVDQU    X9, 16(DI)
	VMOVDQU    X10, 32(DI)
	VMOVDQU    X11, 48(DI)
	ADDQ       DX, DI
	ADDQ       $16, BX
	CMPQ       BX, CX
	JLT        iloop
	RET
