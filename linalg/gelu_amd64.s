#include "textflag.h"

// GELU (erf) times a gate in AVX2, two vectors per iteration (see
// GeluMul): the constants are memory operands, which leaves the 16
// YMM registers to the two lanes. Same operations, in the same order,
// as the portable version.

DATA geluHalf<>+0(SB)/4, $0x3f000000
DATA geluHalf<>+4(SB)/4, $0x3f000000
DATA geluHalf<>+8(SB)/4, $0x3f000000
DATA geluHalf<>+12(SB)/4, $0x3f000000
DATA geluHalf<>+16(SB)/4, $0x3f000000
DATA geluHalf<>+20(SB)/4, $0x3f000000
DATA geluHalf<>+24(SB)/4, $0x3f000000
DATA geluHalf<>+28(SB)/4, $0x3f000000
GLOBL geluHalf<>(SB), RODATA|NOPTR, $32
DATA geluOne<>+0(SB)/4, $0x3f800000
DATA geluOne<>+4(SB)/4, $0x3f800000
DATA geluOne<>+8(SB)/4, $0x3f800000
DATA geluOne<>+12(SB)/4, $0x3f800000
DATA geluOne<>+16(SB)/4, $0x3f800000
DATA geluOne<>+20(SB)/4, $0x3f800000
DATA geluOne<>+24(SB)/4, $0x3f800000
DATA geluOne<>+28(SB)/4, $0x3f800000
GLOBL geluOne<>(SB), RODATA|NOPTR, $32
DATA geluRs2<>+0(SB)/4, $0x3f3504f3
DATA geluRs2<>+4(SB)/4, $0x3f3504f3
DATA geluRs2<>+8(SB)/4, $0x3f3504f3
DATA geluRs2<>+12(SB)/4, $0x3f3504f3
DATA geluRs2<>+16(SB)/4, $0x3f3504f3
DATA geluRs2<>+20(SB)/4, $0x3f3504f3
DATA geluRs2<>+24(SB)/4, $0x3f3504f3
DATA geluRs2<>+28(SB)/4, $0x3f3504f3
GLOBL geluRs2<>(SB), RODATA|NOPTR, $32
DATA geluLo<>+0(SB)/4, $0xc0800000
DATA geluLo<>+4(SB)/4, $0xc0800000
DATA geluLo<>+8(SB)/4, $0xc0800000
DATA geluLo<>+12(SB)/4, $0xc0800000
DATA geluLo<>+16(SB)/4, $0xc0800000
DATA geluLo<>+20(SB)/4, $0xc0800000
DATA geluLo<>+24(SB)/4, $0xc0800000
DATA geluLo<>+28(SB)/4, $0xc0800000
GLOBL geluLo<>(SB), RODATA|NOPTR, $32
DATA geluHi<>+0(SB)/4, $0x40800000
DATA geluHi<>+4(SB)/4, $0x40800000
DATA geluHi<>+8(SB)/4, $0x40800000
DATA geluHi<>+12(SB)/4, $0x40800000
DATA geluHi<>+16(SB)/4, $0x40800000
DATA geluHi<>+20(SB)/4, $0x40800000
DATA geluHi<>+24(SB)/4, $0x40800000
DATA geluHi<>+28(SB)/4, $0x40800000
GLOBL geluHi<>(SB), RODATA|NOPTR, $32
DATA geluA13<>+0(SB)/4, $0xaf95df0b
DATA geluA13<>+4(SB)/4, $0xaf95df0b
DATA geluA13<>+8(SB)/4, $0xaf95df0b
DATA geluA13<>+12(SB)/4, $0xaf95df0b
DATA geluA13<>+16(SB)/4, $0xaf95df0b
DATA geluA13<>+20(SB)/4, $0xaf95df0b
DATA geluA13<>+24(SB)/4, $0xaf95df0b
DATA geluA13<>+28(SB)/4, $0xaf95df0b
GLOBL geluA13<>(SB), RODATA|NOPTR, $32
DATA geluA11<>+0(SB)/4, $0x32edffee
DATA geluA11<>+4(SB)/4, $0x32edffee
DATA geluA11<>+8(SB)/4, $0x32edffee
DATA geluA11<>+12(SB)/4, $0x32edffee
DATA geluA11<>+16(SB)/4, $0x32edffee
DATA geluA11<>+20(SB)/4, $0x32edffee
DATA geluA11<>+24(SB)/4, $0x32edffee
DATA geluA11<>+28(SB)/4, $0x32edffee
GLOBL geluA11<>(SB), RODATA|NOPTR, $32
DATA geluA9<>+0(SB)/4, $0xb60cff51
DATA geluA9<>+4(SB)/4, $0xb60cff51
DATA geluA9<>+8(SB)/4, $0xb60cff51
DATA geluA9<>+12(SB)/4, $0xb60cff51
DATA geluA9<>+16(SB)/4, $0xb60cff51
DATA geluA9<>+20(SB)/4, $0xb60cff51
DATA geluA9<>+24(SB)/4, $0xb60cff51
DATA geluA9<>+28(SB)/4, $0xb60cff51
GLOBL geluA9<>(SB), RODATA|NOPTR, $32
DATA geluA7<>+0(SB)/4, $0xb86ec2d2
DATA geluA7<>+4(SB)/4, $0xb86ec2d2
DATA geluA7<>+8(SB)/4, $0xb86ec2d2
DATA geluA7<>+12(SB)/4, $0xb86ec2d2
DATA geluA7<>+16(SB)/4, $0xb86ec2d2
DATA geluA7<>+20(SB)/4, $0xb86ec2d2
DATA geluA7<>+24(SB)/4, $0xb86ec2d2
DATA geluA7<>+28(SB)/4, $0xb86ec2d2
GLOBL geluA7<>(SB), RODATA|NOPTR, $32
DATA geluA5<>+0(SB)/4, $0xba40ac63
DATA geluA5<>+4(SB)/4, $0xba40ac63
DATA geluA5<>+8(SB)/4, $0xba40ac63
DATA geluA5<>+12(SB)/4, $0xba40ac63
DATA geluA5<>+16(SB)/4, $0xba40ac63
DATA geluA5<>+20(SB)/4, $0xba40ac63
DATA geluA5<>+24(SB)/4, $0xba40ac63
DATA geluA5<>+28(SB)/4, $0xba40ac63
GLOBL geluA5<>(SB), RODATA|NOPTR, $32
DATA geluA3<>+0(SB)/4, $0xbb41a1f6
DATA geluA3<>+4(SB)/4, $0xbb41a1f6
DATA geluA3<>+8(SB)/4, $0xbb41a1f6
DATA geluA3<>+12(SB)/4, $0xbb41a1f6
DATA geluA3<>+16(SB)/4, $0xbb41a1f6
DATA geluA3<>+20(SB)/4, $0xbb41a1f6
DATA geluA3<>+24(SB)/4, $0xbb41a1f6
DATA geluA3<>+28(SB)/4, $0xbb41a1f6
GLOBL geluA3<>(SB), RODATA|NOPTR, $32
DATA geluA1<>+0(SB)/4, $0xbc83dbd4
DATA geluA1<>+4(SB)/4, $0xbc83dbd4
DATA geluA1<>+8(SB)/4, $0xbc83dbd4
DATA geluA1<>+12(SB)/4, $0xbc83dbd4
DATA geluA1<>+16(SB)/4, $0xbc83dbd4
DATA geluA1<>+20(SB)/4, $0xbc83dbd4
DATA geluA1<>+24(SB)/4, $0xbc83dbd4
DATA geluA1<>+28(SB)/4, $0xbc83dbd4
GLOBL geluA1<>(SB), RODATA|NOPTR, $32
DATA geluB8<>+0(SB)/4, $0xb77460cd
DATA geluB8<>+4(SB)/4, $0xb77460cd
DATA geluB8<>+8(SB)/4, $0xb77460cd
DATA geluB8<>+12(SB)/4, $0xb77460cd
DATA geluB8<>+16(SB)/4, $0xb77460cd
DATA geluB8<>+20(SB)/4, $0xb77460cd
DATA geluB8<>+24(SB)/4, $0xb77460cd
DATA geluB8<>+28(SB)/4, $0xb77460cd
GLOBL geluB8<>(SB), RODATA|NOPTR, $32
DATA geluB6<>+0(SB)/4, $0xb95fbd29
DATA geluB6<>+4(SB)/4, $0xb95fbd29
DATA geluB6<>+8(SB)/4, $0xb95fbd29
DATA geluB6<>+12(SB)/4, $0xb95fbd29
DATA geluB6<>+16(SB)/4, $0xb95fbd29
DATA geluB6<>+20(SB)/4, $0xb95fbd29
DATA geluB6<>+24(SB)/4, $0xb95fbd29
DATA geluB6<>+28(SB)/4, $0xb95fbd29
GLOBL geluB6<>(SB), RODATA|NOPTR, $32
DATA geluB4<>+0(SB)/4, $0xbadc924e
DATA geluB4<>+4(SB)/4, $0xbadc924e
DATA geluB4<>+8(SB)/4, $0xbadc924e
DATA geluB4<>+12(SB)/4, $0xbadc924e
DATA geluB4<>+16(SB)/4, $0xbadc924e
DATA geluB4<>+20(SB)/4, $0xbadc924e
DATA geluB4<>+24(SB)/4, $0xbadc924e
DATA geluB4<>+28(SB)/4, $0xbadc924e
GLOBL geluB4<>(SB), RODATA|NOPTR, $32
DATA geluB2<>+0(SB)/4, $0xbbf19bf8
DATA geluB2<>+4(SB)/4, $0xbbf19bf8
DATA geluB2<>+8(SB)/4, $0xbbf19bf8
DATA geluB2<>+12(SB)/4, $0xbbf19bf8
DATA geluB2<>+16(SB)/4, $0xbbf19bf8
DATA geluB2<>+20(SB)/4, $0xbbf19bf8
DATA geluB2<>+24(SB)/4, $0xbbf19bf8
DATA geluB2<>+28(SB)/4, $0xbbf19bf8
GLOBL geluB2<>(SB), RODATA|NOPTR, $32
DATA geluB0<>+0(SB)/4, $0xbc69b6a7
DATA geluB0<>+4(SB)/4, $0xbc69b6a7
DATA geluB0<>+8(SB)/4, $0xbc69b6a7
DATA geluB0<>+12(SB)/4, $0xbc69b6a7
DATA geluB0<>+16(SB)/4, $0xbc69b6a7
DATA geluB0<>+20(SB)/4, $0xbc69b6a7
DATA geluB0<>+24(SB)/4, $0xbc69b6a7
DATA geluB0<>+28(SB)/4, $0xbc69b6a7
GLOBL geluB0<>(SB), RODATA|NOPTR, $32

// func geluMulAVX2(o, a, b *float32, n int)
//
// o[i] = gelu(a[i])·b[i] for n values, a multiple of 16.
TEXT ·geluMulAVX2(SB), NOSPLIT, $0-32
	MOVQ o+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b+16(FP), DX
	MOVQ n+24(FP), CX
	XORQ BX, BX

loop:
	VMOVUPS (SI)(BX*4), Y0
	VMOVUPS 32(SI)(BX*4), Y1
	VMULPS      geluRs2<>(SB), Y0, Y2
	VMAXPS      geluLo<>(SB), Y2, Y2
	VMINPS      geluHi<>(SB), Y2, Y2
	VMULPS      Y2, Y2, Y4
	VMOVUPS     geluA13<>(SB), Y6
	VFMADD213PS geluA11<>(SB), Y4, Y6
	VFMADD213PS geluA9<>(SB), Y4, Y6
	VFMADD213PS geluA7<>(SB), Y4, Y6
	VFMADD213PS geluA5<>(SB), Y4, Y6
	VFMADD213PS geluA3<>(SB), Y4, Y6
	VFMADD213PS geluA1<>(SB), Y4, Y6
	VMULPS      Y2, Y6, Y6
	VMOVUPS     geluB8<>(SB), Y8
	VFMADD213PS geluB6<>(SB), Y4, Y8
	VFMADD213PS geluB4<>(SB), Y4, Y8
	VFMADD213PS geluB2<>(SB), Y4, Y8
	VFMADD213PS geluB0<>(SB), Y4, Y8
	VDIVPS      Y8, Y6, Y6
	VADDPS      geluOne<>(SB), Y6, Y6
	VMULPS      geluHalf<>(SB), Y0, Y0
	VMULPS      Y0, Y6, Y6
	VMULPS      geluRs2<>(SB), Y1, Y3
	VMAXPS      geluLo<>(SB), Y3, Y3
	VMINPS      geluHi<>(SB), Y3, Y3
	VMULPS      Y3, Y3, Y5
	VMOVUPS     geluA13<>(SB), Y7
	VFMADD213PS geluA11<>(SB), Y5, Y7
	VFMADD213PS geluA9<>(SB), Y5, Y7
	VFMADD213PS geluA7<>(SB), Y5, Y7
	VFMADD213PS geluA5<>(SB), Y5, Y7
	VFMADD213PS geluA3<>(SB), Y5, Y7
	VFMADD213PS geluA1<>(SB), Y5, Y7
	VMULPS      Y3, Y7, Y7
	VMOVUPS     geluB8<>(SB), Y9
	VFMADD213PS geluB6<>(SB), Y5, Y9
	VFMADD213PS geluB4<>(SB), Y5, Y9
	VFMADD213PS geluB2<>(SB), Y5, Y9
	VFMADD213PS geluB0<>(SB), Y5, Y9
	VDIVPS      Y9, Y7, Y7
	VADDPS      geluOne<>(SB), Y7, Y7
	VMULPS      geluHalf<>(SB), Y1, Y1
	VMULPS      Y1, Y7, Y7
	VMULPS  (DX)(BX*4), Y6, Y6
	VMULPS  32(DX)(BX*4), Y7, Y7
	VMOVUPS Y6, (DI)(BX*4)
	VMOVUPS Y7, 32(DI)(BX*4)
	ADDQ    $16, BX
	CMPQ    BX, CX
	JLT     loop
	VZEROUPPER
	RET
