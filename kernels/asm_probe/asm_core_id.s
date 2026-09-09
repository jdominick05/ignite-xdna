// Hand-written AIE2 assembly, assembled by Peano's integrated assembler.
// Returns the core's own CORE_ID special register, which the C++ side can check
// against aie::tile::current().global_id().
	.text
	.globl	asm_core_id
	.type	asm_core_id,@function
asm_core_id:
	mov	r0, CORE_ID
	ret	lr
	nop
	nop
	nop
	nop
	nop
.Lfunc_end0:
	.size	asm_core_id, .Lfunc_end0-asm_core_id
