# Génère le micro-noyau AVX-VNNI :
#
#     python3 gen_vnni.py > kernel_vnni_amd64.s
#
# L'assembleur Go n'encode VPDPBUSD qu'en EVEX (AVX-512) ; la forme VEX
# d'AVX-VNNI est donc écrite octet par octet.


def vpdpbusd(acc, a, b):
    # acc += a (uint8, champ vvvv) · b (int8, champ r/m) : VEX.256.66.0F38.W0 50 /r
    R = 0 if acc >= 8 else 1
    B = 0 if b >= 8 else 1
    b1 = (R << 7) | (1 << 6) | (B << 5) | 0x02
    b2 = (((~a) & 0xF) << 3) | (1 << 2) | 0x01
    modrm = 0xC0 | ((acc & 7) << 3) | (b & 7)
    bs = [0xC4, b1, b2, 0x50, modrm]
    return "\tBYTE $0x%02X; BYTE $0x%02X; BYTE $0x%02X; BYTE $0x%02X; BYTE $0x%02X // VPDPBUSD Y%d, Y%d, Y%d" % (*bs, b, a, acc)


out = ['''// Code généré par gen_vnni.py ; NE PAS MODIFIER.

#include "textflag.h"

// func microKernelVNNI(kq int, ap *uint8, lda int, bp *int8, tile *int32)
//
// Tuile 6×16 en entiers : Y0..Y11 accumulent C (ligne r dans Y(2r),
// Y(2r+1)), Y12 et Y13 portent 16 colonnes × 4 profondeurs de B (int8),
// Y14 et Y15 diffusent 4 octets d'une ligne de A (uint8). Les 6 lignes de
// A sont contiguës, espacées de lda octets : SI pointe les lignes 0 à 2,
// R9 les lignes 3 à 5. VPDPBUSD somme les quatre produits de chaque
// colonne dans son accumulateur int32.
TEXT ·microKernelVNNI(SB), NOSPLIT, $0-40
	MOVQ kq+0(FP), CX
	MOVQ ap+8(FP), SI
	MOVQ lda+16(FP), R8
	MOVQ bp+24(FP), DI
	MOVQ tile+32(FP), DX
	LEAQ (SI)(R8*2), R9
	ADDQ R8, R9
''']
for r in range(12):
    out.append("\tVPXOR Y%d, Y%d, Y%d" % (r, r, r))
out.append('''
	TESTQ CX, CX
	JEQ   store

loop:
	VMOVDQU (DI), Y12
	VMOVDQU 32(DI), Y13''')
addr = ["(SI)", "(SI)(R8*1)", "(SI)(R8*2)", "(R9)", "(R9)(R8*1)", "(R9)(R8*2)"]
for r in range(6):
    reg = 14 + (r % 2)
    out.append("\tVPBROADCASTD %s, Y%d" % (addr[r], reg))
    out.append(vpdpbusd(2 * r, reg, 12))
    out.append(vpdpbusd(2 * r + 1, reg, 13))
out.append('''	ADDQ $4, SI
	ADDQ $4, R9
	ADDQ $64, DI
	DECQ CX
	JNZ  loop

store:''')
for r in range(12):
    out.append("\tVMOVDQU Y%d, %s(DX)" % (r, str(32 * r) if r else ""))
out.append("\tVZEROUPPER\n\tRET")
print("\n".join(out))
