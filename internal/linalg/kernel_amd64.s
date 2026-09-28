#include "textflag.h"

// func microKernelAVX2(kb int, ap, bp, tile *float32)
//
// 6x16 tile: Y0..Y11 accumulate C (row r in Y(2r), Y(2r+1)), Y12 and Y13
// carry the 16 values of B at the current depth, Y14 and Y15 the
// broadcasts of A. All 16 YMM registers are used, none spills to the
// stack. VFMADD231PS accumulates in place (dst += a·b).
TEXT ·microKernelAVX2(SB), NOSPLIT, $0-32
	MOVQ kb+0(FP), CX
	MOVQ ap+8(FP), SI
	MOVQ bp+16(FP), DI
	MOVQ tile+24(FP), DX

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	VXORPS Y11, Y11, Y11

	TESTQ CX, CX
	JEQ   store

loop:
	VMOVUPS      (DI), Y12
	VMOVUPS      32(DI), Y13
	VBROADCASTSS (SI), Y14
	VFMADD231PS  Y12, Y14, Y0
	VFMADD231PS  Y13, Y14, Y1
	VBROADCASTSS 4(SI), Y15
	VFMADD231PS  Y12, Y15, Y2
	VFMADD231PS  Y13, Y15, Y3
	VBROADCASTSS 8(SI), Y14
	VFMADD231PS  Y12, Y14, Y4
	VFMADD231PS  Y13, Y14, Y5
	VBROADCASTSS 12(SI), Y15
	VFMADD231PS  Y12, Y15, Y6
	VFMADD231PS  Y13, Y15, Y7
	VBROADCASTSS 16(SI), Y14
	VFMADD231PS  Y12, Y14, Y8
	VFMADD231PS  Y13, Y14, Y9
	VBROADCASTSS 20(SI), Y15
	VFMADD231PS  Y12, Y15, Y10
	VFMADD231PS  Y13, Y15, Y11
	ADDQ         $24, SI
	ADDQ         $64, DI
	DECQ         CX
	JNZ          loop

store:
	VMOVUPS Y0, (DX)
	VMOVUPS Y1, 32(DX)
	VMOVUPS Y2, 64(DX)
	VMOVUPS Y3, 96(DX)
	VMOVUPS Y4, 128(DX)
	VMOVUPS Y5, 160(DX)
	VMOVUPS Y6, 192(DX)
	VMOVUPS Y7, 224(DX)
	VMOVUPS Y8, 256(DX)
	VMOVUPS Y9, 288(DX)
	VMOVUPS Y10, 320(DX)
	VMOVUPS Y11, 352(DX)
	VZEROUPPER
	RET

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET
