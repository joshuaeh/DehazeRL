??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
?
conv2d_transpose_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_29/bias
?
,conv2d_transpose_29/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_29/kernel
?
.conv2d_transpose_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_29/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_28/bias
?
,conv2d_transpose_28/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_28/kernel
?
.conv2d_transpose_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_28/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_27/bias
?
,conv2d_transpose_27/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*+
shared_nameconv2d_transpose_27/kernel
?
.conv2d_transpose_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_27/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_nameconv2d_transpose_26/bias
?
,conv2d_transpose_26/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_26/bias*
_output_shapes
:0*
dtype0
?
conv2d_transpose_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*+
shared_nameconv2d_transpose_26/kernel
?
.conv2d_transpose_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_26/kernel*&
_output_shapes
:0H*
dtype0
?
conv2d_transpose_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*)
shared_nameconv2d_transpose_25/bias
?
,conv2d_transpose_25/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_25/bias*
_output_shapes
:H*
dtype0
?
conv2d_transpose_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*+
shared_nameconv2d_transpose_25/kernel
?
.conv2d_transpose_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_25/kernel*'
_output_shapes
:H?*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:? *
dtype0
{
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H? * 
shared_namedense_31/kernel
t
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes
:	H? *
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op*
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias
 0_jit_compiled_convolution_op*
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op*
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op*
Z
0
1
%2
&3
.4
/5
76
87
@8
A9
I10
J11*
Z
0
1
%2
&3
.4
/5
76
87
@8
A9
I10
J11*
,
L0
M1
N2
O3
P4
Q5* 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_3* 
6
[trace_0
\trace_1
]trace_2
^trace_3* 
* 

_serving_default* 

0
1*

0
1*
	
L0* 
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
_Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_31/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ltrace_0* 

mtrace_0* 

%0
&1*

%0
&1*
	
M0* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
jd
VARIABLE_VALUEconv2d_transpose_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

.0
/1*

.0
/1*
	
N0* 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

ztrace_0* 

{trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_26/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

70
81*

70
81*
	
O0* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_27/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_27/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

@0
A1*

@0
A1*
	
P0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_28/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_28/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

I0
J1*

I0
J1*
	
Q0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_29/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_29/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
* 
<
0
1
2
3
4
5
6
7*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
L0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
	
M0* 
* 
* 
* 
* 
* 
* 
	
N0* 
* 
* 
* 
* 
* 
* 
	
O0* 
* 
* 
* 
* 
* 
* 
	
P0* 
* 
* 
* 
* 
* 
* 
	
Q0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
{
serving_default_input_10Placeholder*'
_output_shapes
:?????????H*
dtype0*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10dense_31/kerneldense_31/biasconv2d_transpose_25/kernelconv2d_transpose_25/biasconv2d_transpose_26/kernelconv2d_transpose_26/biasconv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_transpose_29/kernelconv2d_transpose_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_498768
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp.conv2d_transpose_25/kernel/Read/ReadVariableOp,conv2d_transpose_25/bias/Read/ReadVariableOp.conv2d_transpose_26/kernel/Read/ReadVariableOp,conv2d_transpose_26/bias/Read/ReadVariableOp.conv2d_transpose_27/kernel/Read/ReadVariableOp,conv2d_transpose_27/bias/Read/ReadVariableOp.conv2d_transpose_28/kernel/Read/ReadVariableOp,conv2d_transpose_28/bias/Read/ReadVariableOp.conv2d_transpose_29/kernel/Read/ReadVariableOp,conv2d_transpose_29/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_499861
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_31/kerneldense_31/biasconv2d_transpose_25/kernelconv2d_transpose_25/biasconv2d_transpose_26/kernelconv2d_transpose_26/biasconv2d_transpose_27/kernelconv2d_transpose_27/biasconv2d_transpose_28/kernelconv2d_transpose_28/biasconv2d_transpose_29/kernelconv2d_transpose_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_499907??
?8
?
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_31_layer_call_and_return_conditional_losses_498004

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????? f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?4
?
"__inference__traced_restore_499907
file_prefix3
 assignvariableop_dense_31_kernel:	H? /
 assignvariableop_1_dense_31_bias:	? H
-assignvariableop_2_conv2d_transpose_25_kernel:H?9
+assignvariableop_3_conv2d_transpose_25_bias:HG
-assignvariableop_4_conv2d_transpose_26_kernel:0H9
+assignvariableop_5_conv2d_transpose_26_bias:0G
-assignvariableop_6_conv2d_transpose_27_kernel: 09
+assignvariableop_7_conv2d_transpose_27_bias: G
-assignvariableop_8_conv2d_transpose_28_kernel: 9
+assignvariableop_9_conv2d_transpose_28_bias:H
.assignvariableop_10_conv2d_transpose_29_kernel::
,assignvariableop_11_conv2d_transpose_29_bias:
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_dense_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_25_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_26_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_26_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_27_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_27_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_28_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_28_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv2d_transpose_29_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv2d_transpose_29_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?%
?
__inference__traced_save_499861
file_prefix.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop9
5savev2_conv2d_transpose_25_kernel_read_readvariableop7
3savev2_conv2d_transpose_25_bias_read_readvariableop9
5savev2_conv2d_transpose_26_kernel_read_readvariableop7
3savev2_conv2d_transpose_26_bias_read_readvariableop9
5savev2_conv2d_transpose_27_kernel_read_readvariableop7
3savev2_conv2d_transpose_27_bias_read_readvariableop9
5savev2_conv2d_transpose_28_kernel_read_readvariableop7
3savev2_conv2d_transpose_28_bias_read_readvariableop9
5savev2_conv2d_transpose_29_kernel_read_readvariableop7
3savev2_conv2d_transpose_29_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop5savev2_conv2d_transpose_25_kernel_read_readvariableop3savev2_conv2d_transpose_25_bias_read_readvariableop5savev2_conv2d_transpose_26_kernel_read_readvariableop3savev2_conv2d_transpose_26_bias_read_readvariableop5savev2_conv2d_transpose_27_kernel_read_readvariableop3savev2_conv2d_transpose_27_bias_read_readvariableop5savev2_conv2d_transpose_28_kernel_read_readvariableop3savev2_conv2d_transpose_28_bias_read_readvariableop5savev2_conv2d_transpose_29_kernel_read_readvariableop3savev2_conv2d_transpose_29_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	H? :? :H?:H:0H:0: 0: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	H? :!

_output_shapes	
:? :-)
'
_output_shapes
:H?: 

_output_shapes
:H:,(
&
_output_shapes
:0H: 

_output_shapes
:0:,(
&
_output_shapes
: 0: 

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
(__inference_model_5_layer_call_fn_498887

inputs
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_498142w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_498916

inputs
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_498341w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_27_layer_call_fn_499517

inputs!
unknown: 0
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_498169
input_10
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_498142w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?
?
__inference_loss_fn_2_499742\
Bconv2d_transpose_26_kernel_regularizer_abs_readvariableop_resource:0H
identity??9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_26_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_26_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_26/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp
?
F
*__inference_reshape_5_layer_call_fn_499378

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_499127

inputs:
'dense_31_matmul_readvariableop_resource:	H? 7
(dense_31_biasadd_readvariableop_resource:	? W
<conv2d_transpose_25_conv2d_transpose_readvariableop_resource:H?A
3conv2d_transpose_25_biasadd_readvariableop_resource:HV
<conv2d_transpose_26_conv2d_transpose_readvariableop_resource:0HA
3conv2d_transpose_26_biasadd_readvariableop_resource:0V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource: 0A
3conv2d_transpose_27_biasadd_readvariableop_resource: V
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_28_biasadd_readvariableop_resource:V
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_29_biasadd_readvariableop_resource:
identity??*conv2d_transpose_25/BiasAdd/ReadVariableOp?3conv2d_transpose_25/conv2d_transpose/ReadVariableOp?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_26/BiasAdd/ReadVariableOp?3conv2d_transpose_26/conv2d_transpose/ReadVariableOp?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_27/BiasAdd/ReadVariableOp?3conv2d_transpose_27/conv2d_transpose/ReadVariableOp?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_28/BiasAdd/ReadVariableOp?3conv2d_transpose_28/conv2d_transpose/ReadVariableOp?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_29/BiasAdd/ReadVariableOp?3conv2d_transpose_29/conv2d_transpose/ReadVariableOp?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0|
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? Z
reshape_5/ShapeShapedense_31/Relu:activations:0*
T0*
_output_shapes
:g
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0"reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_5/ReshapeReshapedense_31/Relu:activations:0 reshape_5/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????c
conv2d_transpose_25/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_25/strided_sliceStridedSlice"conv2d_transpose_25/Shape:output:00conv2d_transpose_25/strided_slice/stack:output:02conv2d_transpose_25/strided_slice/stack_1:output:02conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose_25/stackPack*conv2d_transpose_25/strided_slice:output:0$conv2d_transpose_25/stack/1:output:0$conv2d_transpose_25/stack/2:output:0$conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_25/strided_slice_1StridedSlice"conv2d_transpose_25/stack:output:02conv2d_transpose_25/strided_slice_1/stack:output:04conv2d_transpose_25/strided_slice_1/stack_1:output:04conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
$conv2d_transpose_25/conv2d_transposeConv2DBackpropInput"conv2d_transpose_25/stack:output:0;conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0reshape_5/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
*conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose_25/BiasAddBiasAdd-conv2d_transpose_25/conv2d_transpose:output:02conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
conv2d_transpose_25/ReluRelu$conv2d_transpose_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Ho
conv2d_transpose_26/ShapeShape&conv2d_transpose_25/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_26/strided_sliceStridedSlice"conv2d_transpose_26/Shape:output:00conv2d_transpose_26/strided_slice/stack:output:02conv2d_transpose_26/strided_slice/stack_1:output:02conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_26/stackPack*conv2d_transpose_26/strided_slice:output:0$conv2d_transpose_26/stack/1:output:0$conv2d_transpose_26/stack/2:output:0$conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_26/strided_slice_1StridedSlice"conv2d_transpose_26/stack:output:02conv2d_transpose_26/strided_slice_1/stack:output:04conv2d_transpose_26/strided_slice_1/stack_1:output:04conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
$conv2d_transpose_26/conv2d_transposeConv2DBackpropInput"conv2d_transpose_26/stack:output:0;conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_25/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
*conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_26/BiasAddBiasAdd-conv2d_transpose_26/conv2d_transpose:output:02conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
conv2d_transpose_26/ReluRelu$conv2d_transpose_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0o
conv2d_transpose_27/ShapeShape&conv2d_transpose_26/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_26/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_27/ReluRelu$conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? o
conv2d_transpose_28/ShapeShape&conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_28/ReluRelu$conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????o
conv2d_transpose_29/ShapeShape&conv2d_transpose_28/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_28/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_29/SigmoidSigmoid$conv2d_transpose_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: v
IdentityIdentityconv2d_transpose_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp+^conv2d_transpose_25/BiasAdd/ReadVariableOp4^conv2d_transpose_25/conv2d_transpose/ReadVariableOp:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_26/BiasAdd/ReadVariableOp4^conv2d_transpose_26/conv2d_transpose/ReadVariableOp:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_25/BiasAdd/ReadVariableOp*conv2d_transpose_25/BiasAdd/ReadVariableOp2j
3conv2d_transpose_25/conv2d_transpose/ReadVariableOp3conv2d_transpose_25/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_26/BiasAdd/ReadVariableOp*conv2d_transpose_26/BiasAdd/ReadVariableOp2j
3conv2d_transpose_26/conv2d_transpose/ReadVariableOp3conv2d_transpose_26/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_498522
input_10"
dense_31_498400:	H? 
dense_31_498402:	? 5
conv2d_transpose_25_498406:H?(
conv2d_transpose_25_498408:H4
conv2d_transpose_26_498411:0H(
conv2d_transpose_26_498413:04
conv2d_transpose_27_498416: 0(
conv2d_transpose_27_498418: 4
conv2d_transpose_28_498421: (
conv2d_transpose_28_498423:4
conv2d_transpose_29_498426:(
conv2d_transpose_29_498428:
identity??+conv2d_transpose_25/StatefulPartitionedCall?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_26/StatefulPartitionedCall?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_27/StatefulPartitionedCall?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_28/StatefulPartitionedCall?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_29/StatefulPartitionedCall?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp? dense_31/StatefulPartitionedCall?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_31_498400dense_31_498402*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_498004?
reshape_5/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024?
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv2d_transpose_25_498406conv2d_transpose_25_498408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724?
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0conv2d_transpose_26_498411conv2d_transpose_26_498413*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0conv2d_transpose_27_498416conv2d_transpose_27_498418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_498421conv2d_transpose_28_498423*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_transpose_29_498426conv2d_transpose_29_498428*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_31_498400*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_31_498400*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_25_498406*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_25_498406*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_26_498411*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_26_498411*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_27_498416*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_27_498416*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_28_498421*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_28_498421*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_29_498426*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_29_498426*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_25/StatefulPartitionedCall:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_26/StatefulPartitionedCall:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_27/StatefulPartitionedCall:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_28/StatefulPartitionedCall:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_29/StatefulPartitionedCall:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp!^dense_31/StatefulPartitionedCall/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?
?
__inference_loss_fn_3_499762\
Bconv2d_transpose_27_kernel_regularizer_abs_readvariableop_resource: 0
identity??9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_27_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_27_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_27/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp
ڮ
?
C__inference_model_5_layer_call_and_return_conditional_losses_498142

inputs"
dense_31_498005:	H? 
dense_31_498007:	? 5
conv2d_transpose_25_498026:H?(
conv2d_transpose_25_498028:H4
conv2d_transpose_26_498031:0H(
conv2d_transpose_26_498033:04
conv2d_transpose_27_498036: 0(
conv2d_transpose_27_498038: 4
conv2d_transpose_28_498041: (
conv2d_transpose_28_498043:4
conv2d_transpose_29_498046:(
conv2d_transpose_29_498048:
identity??+conv2d_transpose_25/StatefulPartitionedCall?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_26/StatefulPartitionedCall?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_27/StatefulPartitionedCall?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_28/StatefulPartitionedCall?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_29/StatefulPartitionedCall?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp? dense_31/StatefulPartitionedCall?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_498005dense_31_498007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_498004?
reshape_5/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024?
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv2d_transpose_25_498026conv2d_transpose_25_498028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724?
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0conv2d_transpose_26_498031conv2d_transpose_26_498033*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0conv2d_transpose_27_498036conv2d_transpose_27_498038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_498041conv2d_transpose_28_498043*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_transpose_29_498046conv2d_transpose_29_498048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_31_498005*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_31_498005*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_25_498026*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_25_498026*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_26_498031*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_26_498031*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_27_498036*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_27_498036*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_28_498041*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_28_498041*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_29_498046*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_29_498046*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_25/StatefulPartitionedCall:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_26/StatefulPartitionedCall:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_27/StatefulPartitionedCall:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_28/StatefulPartitionedCall:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_29/StatefulPartitionedCall:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp!^dense_31/StatefulPartitionedCall/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
ڮ
?
C__inference_model_5_layer_call_and_return_conditional_losses_498341

inputs"
dense_31_498219:	H? 
dense_31_498221:	? 5
conv2d_transpose_25_498225:H?(
conv2d_transpose_25_498227:H4
conv2d_transpose_26_498230:0H(
conv2d_transpose_26_498232:04
conv2d_transpose_27_498235: 0(
conv2d_transpose_27_498237: 4
conv2d_transpose_28_498240: (
conv2d_transpose_28_498242:4
conv2d_transpose_29_498245:(
conv2d_transpose_29_498247:
identity??+conv2d_transpose_25/StatefulPartitionedCall?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_26/StatefulPartitionedCall?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_27/StatefulPartitionedCall?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_28/StatefulPartitionedCall?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_29/StatefulPartitionedCall?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp? dense_31/StatefulPartitionedCall?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinputsdense_31_498219dense_31_498221*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_498004?
reshape_5/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024?
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv2d_transpose_25_498225conv2d_transpose_25_498227*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724?
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0conv2d_transpose_26_498230conv2d_transpose_26_498232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0conv2d_transpose_27_498235conv2d_transpose_27_498237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_498240conv2d_transpose_28_498242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_transpose_29_498245conv2d_transpose_29_498247*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_31_498219*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_31_498219*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_25_498225*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_25_498225*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_26_498230*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_26_498230*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_27_498235*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_27_498235*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_28_498240*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_28_498240*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_29_498245*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_29_498245*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_25/StatefulPartitionedCall:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_26/StatefulPartitionedCall:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_27/StatefulPartitionedCall:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_28/StatefulPartitionedCall:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_29/StatefulPartitionedCall:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp!^dense_31/StatefulPartitionedCall/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_498647
input_10"
dense_31_498525:	H? 
dense_31_498527:	? 5
conv2d_transpose_25_498531:H?(
conv2d_transpose_25_498533:H4
conv2d_transpose_26_498536:0H(
conv2d_transpose_26_498538:04
conv2d_transpose_27_498541: 0(
conv2d_transpose_27_498543: 4
conv2d_transpose_28_498546: (
conv2d_transpose_28_498548:4
conv2d_transpose_29_498551:(
conv2d_transpose_29_498553:
identity??+conv2d_transpose_25/StatefulPartitionedCall?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_26/StatefulPartitionedCall?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_27/StatefulPartitionedCall?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_28/StatefulPartitionedCall?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_29/StatefulPartitionedCall?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp? dense_31/StatefulPartitionedCall?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
 dense_31/StatefulPartitionedCallStatefulPartitionedCallinput_10dense_31_498525dense_31_498527*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_498004?
reshape_5/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024?
+conv2d_transpose_25/StatefulPartitionedCallStatefulPartitionedCall"reshape_5/PartitionedCall:output:0conv2d_transpose_25_498531conv2d_transpose_25_498533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724?
+conv2d_transpose_26/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_25/StatefulPartitionedCall:output:0conv2d_transpose_26_498536conv2d_transpose_26_498538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784?
+conv2d_transpose_27/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_26/StatefulPartitionedCall:output:0conv2d_transpose_27_498541conv2d_transpose_27_498543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_497844?
+conv2d_transpose_28/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_27/StatefulPartitionedCall:output:0conv2d_transpose_28_498546conv2d_transpose_28_498548*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904?
+conv2d_transpose_29/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_28/StatefulPartitionedCall:output:0conv2d_transpose_29_498551conv2d_transpose_29_498553*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_31_498525*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_31_498525*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_25_498531*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_25_498531*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_26_498536*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_26_498536*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_27_498541*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_27_498541*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_28_498546*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_28_498546*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_29_498551*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_29_498551*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_29/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_25/StatefulPartitionedCall:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_26/StatefulPartitionedCall:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_27/StatefulPartitionedCall:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_28/StatefulPartitionedCall:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_29/StatefulPartitionedCall:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp!^dense_31/StatefulPartitionedCall/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_25/StatefulPartitionedCall+conv2d_transpose_25/StatefulPartitionedCall2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_26/StatefulPartitionedCall+conv2d_transpose_26/StatefulPartitionedCall2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_27/StatefulPartitionedCall+conv2d_transpose_27/StatefulPartitionedCall2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_28/StatefulPartitionedCall+conv2d_transpose_28/StatefulPartitionedCall2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_29/StatefulPartitionedCall+conv2d_transpose_29/StatefulPartitionedCall2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?
?
)__inference_dense_31_layer_call_fn_499347

inputs
unknown:	H? 
	unknown_0:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_498004p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_26_layer_call_fn_499459

inputs!
unknown:0H
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
a
E__inference_reshape_5_layer_call_and_return_conditional_losses_498024

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_499450

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Hy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????H*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????Hj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????Hq
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_28_layer_call_fn_499575

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
E__inference_reshape_5_layer_call_and_return_conditional_losses_499392

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_29_layer_call_fn_499633

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_497964?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
C__inference_model_5_layer_call_and_return_conditional_losses_499338

inputs:
'dense_31_matmul_readvariableop_resource:	H? 7
(dense_31_biasadd_readvariableop_resource:	? W
<conv2d_transpose_25_conv2d_transpose_readvariableop_resource:H?A
3conv2d_transpose_25_biasadd_readvariableop_resource:HV
<conv2d_transpose_26_conv2d_transpose_readvariableop_resource:0HA
3conv2d_transpose_26_biasadd_readvariableop_resource:0V
<conv2d_transpose_27_conv2d_transpose_readvariableop_resource: 0A
3conv2d_transpose_27_biasadd_readvariableop_resource: V
<conv2d_transpose_28_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_28_biasadd_readvariableop_resource:V
<conv2d_transpose_29_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_29_biasadd_readvariableop_resource:
identity??*conv2d_transpose_25/BiasAdd/ReadVariableOp?3conv2d_transpose_25/conv2d_transpose/ReadVariableOp?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_26/BiasAdd/ReadVariableOp?3conv2d_transpose_26/conv2d_transpose/ReadVariableOp?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_27/BiasAdd/ReadVariableOp?3conv2d_transpose_27/conv2d_transpose/ReadVariableOp?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_28/BiasAdd/ReadVariableOp?3conv2d_transpose_28/conv2d_transpose/ReadVariableOp?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_29/BiasAdd/ReadVariableOp?3conv2d_transpose_29/conv2d_transpose/ReadVariableOp?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOp?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0|
dense_31/MatMulMatMulinputs&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_31/ReluReludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? Z
reshape_5/ShapeShapedense_31/Relu:activations:0*
T0*
_output_shapes
:g
reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_5/strided_sliceStridedSlicereshape_5/Shape:output:0&reshape_5/strided_slice/stack:output:0(reshape_5/strided_slice/stack_1:output:0(reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape_5/Reshape/shapePack reshape_5/strided_slice:output:0"reshape_5/Reshape/shape/1:output:0"reshape_5/Reshape/shape/2:output:0"reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_5/ReshapeReshapedense_31/Relu:activations:0 reshape_5/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????c
conv2d_transpose_25/ShapeShapereshape_5/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_25/strided_sliceStridedSlice"conv2d_transpose_25/Shape:output:00conv2d_transpose_25/strided_slice/stack:output:02conv2d_transpose_25/strided_slice/stack_1:output:02conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose_25/stackPack*conv2d_transpose_25/strided_slice:output:0$conv2d_transpose_25/stack/1:output:0$conv2d_transpose_25/stack/2:output:0$conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_25/strided_slice_1StridedSlice"conv2d_transpose_25/stack:output:02conv2d_transpose_25/strided_slice_1/stack:output:04conv2d_transpose_25/strided_slice_1/stack_1:output:04conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
$conv2d_transpose_25/conv2d_transposeConv2DBackpropInput"conv2d_transpose_25/stack:output:0;conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0reshape_5/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
*conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose_25/BiasAddBiasAdd-conv2d_transpose_25/conv2d_transpose:output:02conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
conv2d_transpose_25/ReluRelu$conv2d_transpose_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Ho
conv2d_transpose_26/ShapeShape&conv2d_transpose_25/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_26/strided_sliceStridedSlice"conv2d_transpose_26/Shape:output:00conv2d_transpose_26/strided_slice/stack:output:02conv2d_transpose_26/strided_slice/stack_1:output:02conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_26/stackPack*conv2d_transpose_26/strided_slice:output:0$conv2d_transpose_26/stack/1:output:0$conv2d_transpose_26/stack/2:output:0$conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_26/strided_slice_1StridedSlice"conv2d_transpose_26/stack:output:02conv2d_transpose_26/strided_slice_1/stack:output:04conv2d_transpose_26/strided_slice_1/stack_1:output:04conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
$conv2d_transpose_26/conv2d_transposeConv2DBackpropInput"conv2d_transpose_26/stack:output:0;conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_25/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
*conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_26/BiasAddBiasAdd-conv2d_transpose_26/conv2d_transpose:output:02conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
conv2d_transpose_26/ReluRelu$conv2d_transpose_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0o
conv2d_transpose_27/ShapeShape&conv2d_transpose_26/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_27/strided_sliceStridedSlice"conv2d_transpose_27/Shape:output:00conv2d_transpose_27/strided_slice/stack:output:02conv2d_transpose_27/strided_slice/stack_1:output:02conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_27/stackPack*conv2d_transpose_27/strided_slice:output:0$conv2d_transpose_27/stack/1:output:0$conv2d_transpose_27/stack/2:output:0$conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_27/strided_slice_1StridedSlice"conv2d_transpose_27/stack:output:02conv2d_transpose_27/strided_slice_1/stack:output:04conv2d_transpose_27/strided_slice_1/stack_1:output:04conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
$conv2d_transpose_27/conv2d_transposeConv2DBackpropInput"conv2d_transpose_27/stack:output:0;conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_26/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_27/BiasAddBiasAdd-conv2d_transpose_27/conv2d_transpose:output:02conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_27/ReluRelu$conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? o
conv2d_transpose_28/ShapeShape&conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_28/strided_sliceStridedSlice"conv2d_transpose_28/Shape:output:00conv2d_transpose_28/strided_slice/stack:output:02conv2d_transpose_28/strided_slice/stack_1:output:02conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_28/stackPack*conv2d_transpose_28/strided_slice:output:0$conv2d_transpose_28/stack/1:output:0$conv2d_transpose_28/stack/2:output:0$conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_28/strided_slice_1StridedSlice"conv2d_transpose_28/stack:output:02conv2d_transpose_28/strided_slice_1/stack:output:04conv2d_transpose_28/strided_slice_1/stack_1:output:04conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_28/conv2d_transposeConv2DBackpropInput"conv2d_transpose_28/stack:output:0;conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_28/BiasAddBiasAdd-conv2d_transpose_28/conv2d_transpose:output:02conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_28/ReluRelu$conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????o
conv2d_transpose_29/ShapeShape&conv2d_transpose_28/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_29/strided_sliceStridedSlice"conv2d_transpose_29/Shape:output:00conv2d_transpose_29/strided_slice/stack:output:02conv2d_transpose_29/strided_slice/stack_1:output:02conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_29/stackPack*conv2d_transpose_29/strided_slice:output:0$conv2d_transpose_29/stack/1:output:0$conv2d_transpose_29/stack/2:output:0$conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_29/strided_slice_1StridedSlice"conv2d_transpose_29/stack:output:02conv2d_transpose_29/strided_slice_1/stack:output:04conv2d_transpose_29/strided_slice_1/stack_1:output:04conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_29/conv2d_transposeConv2DBackpropInput"conv2d_transpose_29/stack:output:0;conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_28/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_29/BiasAddBiasAdd-conv2d_transpose_29/conv2d_transpose:output:02conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_29/SigmoidSigmoid$conv2d_transpose_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: v
IdentityIdentityconv2d_transpose_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp+^conv2d_transpose_25/BiasAdd/ReadVariableOp4^conv2d_transpose_25/conv2d_transpose/ReadVariableOp:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_26/BiasAdd/ReadVariableOp4^conv2d_transpose_26/conv2d_transpose/ReadVariableOp:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_27/BiasAdd/ReadVariableOp4^conv2d_transpose_27/conv2d_transpose/ReadVariableOp:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_28/BiasAdd/ReadVariableOp4^conv2d_transpose_28/conv2d_transpose/ReadVariableOp:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_29/BiasAdd/ReadVariableOp4^conv2d_transpose_29/conv2d_transpose/ReadVariableOp:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_25/BiasAdd/ReadVariableOp*conv2d_transpose_25/BiasAdd/ReadVariableOp2j
3conv2d_transpose_25/conv2d_transpose/ReadVariableOp3conv2d_transpose_25/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_26/BiasAdd/ReadVariableOp*conv2d_transpose_26/BiasAdd/ReadVariableOp2j
3conv2d_transpose_26/conv2d_transpose/ReadVariableOp3conv2d_transpose_26/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_27/BiasAdd/ReadVariableOp*conv2d_transpose_27/BiasAdd/ReadVariableOp2j
3conv2d_transpose_27/conv2d_transpose/ReadVariableOp3conv2d_transpose_27/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_28/BiasAdd/ReadVariableOp*conv2d_transpose_28/BiasAdd/ReadVariableOp2j
3conv2d_transpose_28/conv2d_transpose/ReadVariableOp3conv2d_transpose_28/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_29/BiasAdd/ReadVariableOp*conv2d_transpose_29/BiasAdd/ReadVariableOp2j
3conv2d_transpose_29/conv2d_transpose/ReadVariableOp3conv2d_transpose_29/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_499508

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????0q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :Hy
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????H*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????Hj
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????Hq
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_499722]
Bconv2d_transpose_25_kernel_regularizer_abs_readvariableop_resource:H?
identity??9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_25_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_25/kernel/Regularizer/AbsAbsAconv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_25/kernel/Regularizer/SumSum.conv2d_transpose_25/kernel/Regularizer/Abs:y:07conv2d_transpose_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_25/kernel/Regularizer/mulMul5conv2d_transpose_25/kernel/Regularizer/mul/x:output:03conv2d_transpose_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_25/kernel/Regularizer/addAddV25conv2d_transpose_25/kernel/Regularizer/Const:output:0.conv2d_transpose_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_25_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_25/kernel/Regularizer/SquareSquareDconv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_25/kernel/Regularizer/Sum_1Sum1conv2d_transpose_25/kernel/Regularizer/Square:y:07conv2d_transpose_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_25/kernel/Regularizer/mul_1Mul7conv2d_transpose_25/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_25/kernel/Regularizer/add_1AddV2.conv2d_transpose_25/kernel/Regularizer/add:z:00conv2d_transpose_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_25/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_25/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_25/kernel/Regularizer/Square/ReadVariableOp
?8
?
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_499566

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? q
,conv2d_transpose_27/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_27/kernel/Regularizer/AbsAbsAconv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_27/kernel/Regularizer/SumSum.conv2d_transpose_27/kernel/Regularizer/Abs:y:07conv2d_transpose_27/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_27/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_27/kernel/Regularizer/mulMul5conv2d_transpose_27/kernel/Regularizer/mul/x:output:03conv2d_transpose_27/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_27/kernel/Regularizer/addAddV25conv2d_transpose_27/kernel/Regularizer/Const:output:0.conv2d_transpose_27/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_27/kernel/Regularizer/SquareSquareDconv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_27/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_27/kernel/Regularizer/Sum_1Sum1conv2d_transpose_27/kernel/Regularizer/Square:y:07conv2d_transpose_27/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_27/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_27/kernel/Regularizer/mul_1Mul7conv2d_transpose_27/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_27/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_27/kernel/Regularizer/add_1AddV2.conv2d_transpose_27/kernel/Regularizer/add:z:00conv2d_transpose_27/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_27/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_27/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_499782\
Bconv2d_transpose_28_kernel_regularizer_abs_readvariableop_resource: 
identity??9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_28_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_28_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_28/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp
?8
?
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_499682

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_499702J
7dense_31_kernel_regularizer_abs_readvariableop_resource:	H? 
identity??.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOpf
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_31_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_31_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_31/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp
?8
?
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_497904

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_25_layer_call_fn_499401

inputs"
unknown:H?
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_497724?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_497784

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :0y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????0q
,conv2d_transpose_26/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_26/kernel/Regularizer/AbsAbsAconv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_26/kernel/Regularizer/SumSum.conv2d_transpose_26/kernel/Regularizer/Abs:y:07conv2d_transpose_26/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_26/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_26/kernel/Regularizer/mulMul5conv2d_transpose_26/kernel/Regularizer/mul/x:output:03conv2d_transpose_26/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_26/kernel/Regularizer/addAddV25conv2d_transpose_26/kernel/Regularizer/Const:output:0.conv2d_transpose_26/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_26/kernel/Regularizer/SquareSquareDconv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_26/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_26/kernel/Regularizer/Sum_1Sum1conv2d_transpose_26/kernel/Regularizer/Square:y:07conv2d_transpose_26/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_26/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_26/kernel/Regularizer/mul_1Mul7conv2d_transpose_26/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_26/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_26/kernel/Regularizer/add_1AddV2.conv2d_transpose_26/kernel/Regularizer/add:z:00conv2d_transpose_26/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_26/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_26/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
(__inference_model_5_layer_call_fn_498397
input_10
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_5_layer_call_and_return_conditional_losses_498341w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?
?
$__inference_signature_wrapper_498768
input_10
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_497671w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?8
?
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_499624

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????q
,conv2d_transpose_28/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_28/kernel/Regularizer/AbsAbsAconv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_28/kernel/Regularizer/SumSum.conv2d_transpose_28/kernel/Regularizer/Abs:y:07conv2d_transpose_28/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_28/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_28/kernel/Regularizer/mulMul5conv2d_transpose_28/kernel/Regularizer/mul/x:output:03conv2d_transpose_28/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_28/kernel/Regularizer/addAddV25conv2d_transpose_28/kernel/Regularizer/Const:output:0.conv2d_transpose_28/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_28/kernel/Regularizer/SquareSquareDconv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_28/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_28/kernel/Regularizer/Sum_1Sum1conv2d_transpose_28/kernel/Regularizer/Square:y:07conv2d_transpose_28/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_28/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_28/kernel/Regularizer/mul_1Mul7conv2d_transpose_28/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_28/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_28/kernel/Regularizer/add_1AddV2.conv2d_transpose_28/kernel/Regularizer/add:z:00conv2d_transpose_28/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_28/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_28/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ӱ
?
!__inference__wrapped_model_497671
input_10B
/model_5_dense_31_matmul_readvariableop_resource:	H? ?
0model_5_dense_31_biasadd_readvariableop_resource:	? _
Dmodel_5_conv2d_transpose_25_conv2d_transpose_readvariableop_resource:H?I
;model_5_conv2d_transpose_25_biasadd_readvariableop_resource:H^
Dmodel_5_conv2d_transpose_26_conv2d_transpose_readvariableop_resource:0HI
;model_5_conv2d_transpose_26_biasadd_readvariableop_resource:0^
Dmodel_5_conv2d_transpose_27_conv2d_transpose_readvariableop_resource: 0I
;model_5_conv2d_transpose_27_biasadd_readvariableop_resource: ^
Dmodel_5_conv2d_transpose_28_conv2d_transpose_readvariableop_resource: I
;model_5_conv2d_transpose_28_biasadd_readvariableop_resource:^
Dmodel_5_conv2d_transpose_29_conv2d_transpose_readvariableop_resource:I
;model_5_conv2d_transpose_29_biasadd_readvariableop_resource:
identity??2model_5/conv2d_transpose_25/BiasAdd/ReadVariableOp?;model_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOp?2model_5/conv2d_transpose_26/BiasAdd/ReadVariableOp?;model_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOp?2model_5/conv2d_transpose_27/BiasAdd/ReadVariableOp?;model_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOp?2model_5/conv2d_transpose_28/BiasAdd/ReadVariableOp?;model_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOp?2model_5/conv2d_transpose_29/BiasAdd/ReadVariableOp?;model_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOp?'model_5/dense_31/BiasAdd/ReadVariableOp?&model_5/dense_31/MatMul/ReadVariableOp?
&model_5/dense_31/MatMul/ReadVariableOpReadVariableOp/model_5_dense_31_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
model_5/dense_31/MatMulMatMulinput_10.model_5/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
'model_5/dense_31/BiasAdd/ReadVariableOpReadVariableOp0model_5_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
model_5/dense_31/BiasAddBiasAdd!model_5/dense_31/MatMul:product:0/model_5/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
model_5/dense_31/ReluRelu!model_5/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? j
model_5/reshape_5/ShapeShape#model_5/dense_31/Relu:activations:0*
T0*
_output_shapes
:o
%model_5/reshape_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_5/reshape_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_5/reshape_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_5/reshape_5/strided_sliceStridedSlice model_5/reshape_5/Shape:output:0.model_5/reshape_5/strided_slice/stack:output:00model_5/reshape_5/strided_slice/stack_1:output:00model_5/reshape_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_5/reshape_5/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_5/reshape_5/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
!model_5/reshape_5/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
model_5/reshape_5/Reshape/shapePack(model_5/reshape_5/strided_slice:output:0*model_5/reshape_5/Reshape/shape/1:output:0*model_5/reshape_5/Reshape/shape/2:output:0*model_5/reshape_5/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_5/reshape_5/ReshapeReshape#model_5/dense_31/Relu:activations:0(model_5/reshape_5/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????s
!model_5/conv2d_transpose_25/ShapeShape"model_5/reshape_5/Reshape:output:0*
T0*
_output_shapes
:y
/model_5/conv2d_transpose_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_5/conv2d_transpose_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_5/conv2d_transpose_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_5/conv2d_transpose_25/strided_sliceStridedSlice*model_5/conv2d_transpose_25/Shape:output:08model_5/conv2d_transpose_25/strided_slice/stack:output:0:model_5/conv2d_transpose_25/strided_slice/stack_1:output:0:model_5/conv2d_transpose_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_5/conv2d_transpose_25/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_25/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_25/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
!model_5/conv2d_transpose_25/stackPack2model_5/conv2d_transpose_25/strided_slice:output:0,model_5/conv2d_transpose_25/stack/1:output:0,model_5/conv2d_transpose_25/stack/2:output:0,model_5/conv2d_transpose_25/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_5/conv2d_transpose_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_5/conv2d_transpose_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_5/conv2d_transpose_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_5/conv2d_transpose_25/strided_slice_1StridedSlice*model_5/conv2d_transpose_25/stack:output:0:model_5/conv2d_transpose_25/strided_slice_1/stack:output:0<model_5/conv2d_transpose_25/strided_slice_1/stack_1:output:0<model_5/conv2d_transpose_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_5_conv2d_transpose_25_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
,model_5/conv2d_transpose_25/conv2d_transposeConv2DBackpropInput*model_5/conv2d_transpose_25/stack:output:0Cmodel_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOp:value:0"model_5/reshape_5/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
2model_5/conv2d_transpose_25/BiasAdd/ReadVariableOpReadVariableOp;model_5_conv2d_transpose_25_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
#model_5/conv2d_transpose_25/BiasAddBiasAdd5model_5/conv2d_transpose_25/conv2d_transpose:output:0:model_5/conv2d_transpose_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
 model_5/conv2d_transpose_25/ReluRelu,model_5/conv2d_transpose_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H
!model_5/conv2d_transpose_26/ShapeShape.model_5/conv2d_transpose_25/Relu:activations:0*
T0*
_output_shapes
:y
/model_5/conv2d_transpose_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_5/conv2d_transpose_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_5/conv2d_transpose_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_5/conv2d_transpose_26/strided_sliceStridedSlice*model_5/conv2d_transpose_26/Shape:output:08model_5/conv2d_transpose_26/strided_slice/stack:output:0:model_5/conv2d_transpose_26/strided_slice/stack_1:output:0:model_5/conv2d_transpose_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_5/conv2d_transpose_26/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_26/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_26/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
!model_5/conv2d_transpose_26/stackPack2model_5/conv2d_transpose_26/strided_slice:output:0,model_5/conv2d_transpose_26/stack/1:output:0,model_5/conv2d_transpose_26/stack/2:output:0,model_5/conv2d_transpose_26/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_5/conv2d_transpose_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_5/conv2d_transpose_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_5/conv2d_transpose_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_5/conv2d_transpose_26/strided_slice_1StridedSlice*model_5/conv2d_transpose_26/stack:output:0:model_5/conv2d_transpose_26/strided_slice_1/stack:output:0<model_5/conv2d_transpose_26/strided_slice_1/stack_1:output:0<model_5/conv2d_transpose_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_5_conv2d_transpose_26_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,model_5/conv2d_transpose_26/conv2d_transposeConv2DBackpropInput*model_5/conv2d_transpose_26/stack:output:0Cmodel_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOp:value:0.model_5/conv2d_transpose_25/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
2model_5/conv2d_transpose_26/BiasAdd/ReadVariableOpReadVariableOp;model_5_conv2d_transpose_26_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#model_5/conv2d_transpose_26/BiasAddBiasAdd5model_5/conv2d_transpose_26/conv2d_transpose:output:0:model_5/conv2d_transpose_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
 model_5/conv2d_transpose_26/ReluRelu,model_5/conv2d_transpose_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0
!model_5/conv2d_transpose_27/ShapeShape.model_5/conv2d_transpose_26/Relu:activations:0*
T0*
_output_shapes
:y
/model_5/conv2d_transpose_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_5/conv2d_transpose_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_5/conv2d_transpose_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_5/conv2d_transpose_27/strided_sliceStridedSlice*model_5/conv2d_transpose_27/Shape:output:08model_5/conv2d_transpose_27/strided_slice/stack:output:0:model_5/conv2d_transpose_27/strided_slice/stack_1:output:0:model_5/conv2d_transpose_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_5/conv2d_transpose_27/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_27/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_27/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
!model_5/conv2d_transpose_27/stackPack2model_5/conv2d_transpose_27/strided_slice:output:0,model_5/conv2d_transpose_27/stack/1:output:0,model_5/conv2d_transpose_27/stack/2:output:0,model_5/conv2d_transpose_27/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_5/conv2d_transpose_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_5/conv2d_transpose_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_5/conv2d_transpose_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_5/conv2d_transpose_27/strided_slice_1StridedSlice*model_5/conv2d_transpose_27/stack:output:0:model_5/conv2d_transpose_27/strided_slice_1/stack:output:0<model_5/conv2d_transpose_27/strided_slice_1/stack_1:output:0<model_5/conv2d_transpose_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_5_conv2d_transpose_27_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,model_5/conv2d_transpose_27/conv2d_transposeConv2DBackpropInput*model_5/conv2d_transpose_27/stack:output:0Cmodel_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOp:value:0.model_5/conv2d_transpose_26/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2model_5/conv2d_transpose_27/BiasAdd/ReadVariableOpReadVariableOp;model_5_conv2d_transpose_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#model_5/conv2d_transpose_27/BiasAddBiasAdd5model_5/conv2d_transpose_27/conv2d_transpose:output:0:model_5/conv2d_transpose_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 model_5/conv2d_transpose_27/ReluRelu,model_5/conv2d_transpose_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
!model_5/conv2d_transpose_28/ShapeShape.model_5/conv2d_transpose_27/Relu:activations:0*
T0*
_output_shapes
:y
/model_5/conv2d_transpose_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_5/conv2d_transpose_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_5/conv2d_transpose_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_5/conv2d_transpose_28/strided_sliceStridedSlice*model_5/conv2d_transpose_28/Shape:output:08model_5/conv2d_transpose_28/strided_slice/stack:output:0:model_5/conv2d_transpose_28/strided_slice/stack_1:output:0:model_5/conv2d_transpose_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_5/conv2d_transpose_28/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_28/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_28/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
!model_5/conv2d_transpose_28/stackPack2model_5/conv2d_transpose_28/strided_slice:output:0,model_5/conv2d_transpose_28/stack/1:output:0,model_5/conv2d_transpose_28/stack/2:output:0,model_5/conv2d_transpose_28/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_5/conv2d_transpose_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_5/conv2d_transpose_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_5/conv2d_transpose_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_5/conv2d_transpose_28/strided_slice_1StridedSlice*model_5/conv2d_transpose_28/stack:output:0:model_5/conv2d_transpose_28/strided_slice_1/stack:output:0<model_5/conv2d_transpose_28/strided_slice_1/stack_1:output:0<model_5/conv2d_transpose_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_5_conv2d_transpose_28_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,model_5/conv2d_transpose_28/conv2d_transposeConv2DBackpropInput*model_5/conv2d_transpose_28/stack:output:0Cmodel_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOp:value:0.model_5/conv2d_transpose_27/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
2model_5/conv2d_transpose_28/BiasAdd/ReadVariableOpReadVariableOp;model_5_conv2d_transpose_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#model_5/conv2d_transpose_28/BiasAddBiasAdd5model_5/conv2d_transpose_28/conv2d_transpose:output:0:model_5/conv2d_transpose_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model_5/conv2d_transpose_28/ReluRelu,model_5/conv2d_transpose_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
!model_5/conv2d_transpose_29/ShapeShape.model_5/conv2d_transpose_28/Relu:activations:0*
T0*
_output_shapes
:y
/model_5/conv2d_transpose_29/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_5/conv2d_transpose_29/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_5/conv2d_transpose_29/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_5/conv2d_transpose_29/strided_sliceStridedSlice*model_5/conv2d_transpose_29/Shape:output:08model_5/conv2d_transpose_29/strided_slice/stack:output:0:model_5/conv2d_transpose_29/strided_slice/stack_1:output:0:model_5/conv2d_transpose_29/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_5/conv2d_transpose_29/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_29/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_5/conv2d_transpose_29/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
!model_5/conv2d_transpose_29/stackPack2model_5/conv2d_transpose_29/strided_slice:output:0,model_5/conv2d_transpose_29/stack/1:output:0,model_5/conv2d_transpose_29/stack/2:output:0,model_5/conv2d_transpose_29/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_5/conv2d_transpose_29/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_5/conv2d_transpose_29/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_5/conv2d_transpose_29/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_5/conv2d_transpose_29/strided_slice_1StridedSlice*model_5/conv2d_transpose_29/stack:output:0:model_5/conv2d_transpose_29/strided_slice_1/stack:output:0<model_5/conv2d_transpose_29/strided_slice_1/stack_1:output:0<model_5/conv2d_transpose_29/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_5_conv2d_transpose_29_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,model_5/conv2d_transpose_29/conv2d_transposeConv2DBackpropInput*model_5/conv2d_transpose_29/stack:output:0Cmodel_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOp:value:0.model_5/conv2d_transpose_28/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
2model_5/conv2d_transpose_29/BiasAdd/ReadVariableOpReadVariableOp;model_5_conv2d_transpose_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#model_5/conv2d_transpose_29/BiasAddBiasAdd5model_5/conv2d_transpose_29/conv2d_transpose:output:0:model_5/conv2d_transpose_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
#model_5/conv2d_transpose_29/SigmoidSigmoid,model_5/conv2d_transpose_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????~
IdentityIdentity'model_5/conv2d_transpose_29/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^model_5/conv2d_transpose_25/BiasAdd/ReadVariableOp<^model_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOp3^model_5/conv2d_transpose_26/BiasAdd/ReadVariableOp<^model_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOp3^model_5/conv2d_transpose_27/BiasAdd/ReadVariableOp<^model_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOp3^model_5/conv2d_transpose_28/BiasAdd/ReadVariableOp<^model_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOp3^model_5/conv2d_transpose_29/BiasAdd/ReadVariableOp<^model_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOp(^model_5/dense_31/BiasAdd/ReadVariableOp'^model_5/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2h
2model_5/conv2d_transpose_25/BiasAdd/ReadVariableOp2model_5/conv2d_transpose_25/BiasAdd/ReadVariableOp2z
;model_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOp;model_5/conv2d_transpose_25/conv2d_transpose/ReadVariableOp2h
2model_5/conv2d_transpose_26/BiasAdd/ReadVariableOp2model_5/conv2d_transpose_26/BiasAdd/ReadVariableOp2z
;model_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOp;model_5/conv2d_transpose_26/conv2d_transpose/ReadVariableOp2h
2model_5/conv2d_transpose_27/BiasAdd/ReadVariableOp2model_5/conv2d_transpose_27/BiasAdd/ReadVariableOp2z
;model_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOp;model_5/conv2d_transpose_27/conv2d_transpose/ReadVariableOp2h
2model_5/conv2d_transpose_28/BiasAdd/ReadVariableOp2model_5/conv2d_transpose_28/BiasAdd/ReadVariableOp2z
;model_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOp;model_5/conv2d_transpose_28/conv2d_transpose/ReadVariableOp2h
2model_5/conv2d_transpose_29/BiasAdd/ReadVariableOp2model_5/conv2d_transpose_29/BiasAdd/ReadVariableOp2z
;model_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOp;model_5/conv2d_transpose_29/conv2d_transpose/ReadVariableOp2R
'model_5/dense_31/BiasAdd/ReadVariableOp'model_5/dense_31/BiasAdd/ReadVariableOp2P
&model_5/dense_31/MatMul/ReadVariableOp&model_5/dense_31/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????H
"
_user_specified_name
input_10
?
?
__inference_loss_fn_5_499802\
Bconv2d_transpose_29_kernel_regularizer_abs_readvariableop_resource:
identity??9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_29/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_29_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_29/kernel/Regularizer/AbsAbsAconv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_29/kernel/Regularizer/SumSum.conv2d_transpose_29/kernel/Regularizer/Abs:y:07conv2d_transpose_29/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_29/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_29/kernel/Regularizer/mulMul5conv2d_transpose_29/kernel/Regularizer/mul/x:output:03conv2d_transpose_29/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_29/kernel/Regularizer/addAddV25conv2d_transpose_29/kernel/Regularizer/Const:output:0.conv2d_transpose_29/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_29_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_29/kernel/Regularizer/SquareSquareDconv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_29/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_29/kernel/Regularizer/Sum_1Sum1conv2d_transpose_29/kernel/Regularizer/Square:y:07conv2d_transpose_29/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_29/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_29/kernel/Regularizer/mul_1Mul7conv2d_transpose_29/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_29/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_29/kernel/Regularizer/add_1AddV2.conv2d_transpose_29/kernel/Regularizer/add:z:00conv2d_transpose_29/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_29/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_29/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_29/kernel/Regularizer/Square/ReadVariableOp
?
?
D__inference_dense_31_layer_call_and_return_conditional_losses_499373

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_31/kernel/Regularizer/Abs/ReadVariableOp?1dense_31/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????? f
!dense_31/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_31/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_31/kernel/Regularizer/AbsAbs6dense_31/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_31/kernel/Regularizer/SumSum#dense_31/kernel/Regularizer/Abs:y:0,dense_31/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_31/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_31/kernel/Regularizer/mulMul*dense_31/kernel/Regularizer/mul/x:output:0(dense_31/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_31/kernel/Regularizer/addAddV2*dense_31/kernel/Regularizer/Const:output:0#dense_31/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_31/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_31/kernel/Regularizer/SquareSquare9dense_31/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_31/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_31/kernel/Regularizer/Sum_1Sum&dense_31/kernel/Regularizer/Square:y:0,dense_31/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_31/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_31/kernel/Regularizer/mul_1Mul,dense_31/kernel/Regularizer/mul_1/x:output:0*dense_31/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_31/kernel/Regularizer/add_1AddV2#dense_31/kernel/Regularizer/add:z:0%dense_31/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_31/kernel/Regularizer/Abs/ReadVariableOp2^dense_31/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_31/kernel/Regularizer/Abs/ReadVariableOp.dense_31/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_31/kernel/Regularizer/Square/ReadVariableOp1dense_31/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_101
serving_default_input_10:0?????????HO
conv2d_transpose_298
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias
 '_jit_compiled_convolution_op"
_tf_keras_layer
?
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

.kernel
/bias
 0_jit_compiled_convolution_op"
_tf_keras_layer
?
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op"
_tf_keras_layer
?
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias
 K_jit_compiled_convolution_op"
_tf_keras_layer
v
0
1
%2
&3
.4
/5
76
87
@8
A9
I10
J11"
trackable_list_wrapper
v
0
1
%2
&3
.4
/5
76
87
@8
A9
I10
J11"
trackable_list_wrapper
J
L0
M1
N2
O3
P4
Q5"
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_32?
(__inference_model_5_layer_call_fn_498169
(__inference_model_5_layer_call_fn_498887
(__inference_model_5_layer_call_fn_498916
(__inference_model_5_layer_call_fn_498397?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 zWtrace_0zXtrace_1zYtrace_2zZtrace_3
?
[trace_0
\trace_1
]trace_2
^trace_32?
C__inference_model_5_layer_call_and_return_conditional_losses_499127
C__inference_model_5_layer_call_and_return_conditional_losses_499338
C__inference_model_5_layer_call_and_return_conditional_losses_498522
C__inference_model_5_layer_call_and_return_conditional_losses_498647?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z[trace_0z\trace_1z]trace_2z^trace_3
?B?
!__inference__wrapped_model_497671input_10"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
_serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
?
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
etrace_02?
)__inference_dense_31_layer_call_fn_499347?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zetrace_0
?
ftrace_02?
D__inference_dense_31_layer_call_and_return_conditional_losses_499373?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zftrace_0
": 	H? 2dense_31/kernel
:? 2dense_31/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
ltrace_02?
*__inference_reshape_5_layer_call_fn_499378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zltrace_0
?
mtrace_02?
E__inference_reshape_5_layer_call_and_return_conditional_losses_499392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zmtrace_0
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?
strace_02?
4__inference_conv2d_transpose_25_layer_call_fn_499401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zstrace_0
?
ttrace_02?
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_499450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zttrace_0
5:3H?2conv2d_transpose_25/kernel
&:$H2conv2d_transpose_25/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
?
ztrace_02?
4__inference_conv2d_transpose_26_layer_call_fn_499459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 zztrace_0
?
{trace_02?
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_499508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z{trace_0
4:20H2conv2d_transpose_26/kernel
&:$02conv2d_transpose_26/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
4__inference_conv2d_transpose_27_layer_call_fn_499517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_499566?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2 02conv2d_transpose_27/kernel
&:$ 2conv2d_transpose_27/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
'
P0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
4__inference_conv2d_transpose_28_layer_call_fn_499575?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_499624?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2 2conv2d_transpose_28/kernel
&:$2conv2d_transpose_28/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
4__inference_conv2d_transpose_29_layer_call_fn_499633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_499682?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:22conv2d_transpose_29/kernel
&:$2conv2d_transpose_29/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
?
?trace_02?
__inference_loss_fn_0_499702?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_1_499722?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_2_499742?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_3_499762?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_4_499782?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
?
?trace_02?
__inference_loss_fn_5_499802?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? z?trace_0
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_model_5_layer_call_fn_498169input_10"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_5_layer_call_fn_498887inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_5_layer_call_fn_498916inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
(__inference_model_5_layer_call_fn_498397input_10"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_5_layer_call_and_return_conditional_losses_499127inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_5_layer_call_and_return_conditional_losses_499338inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_5_layer_call_and_return_conditional_losses_498522input_10"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
C__inference_model_5_layer_call_and_return_conditional_losses_498647input_10"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_498768input_10"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_dense_31_layer_call_fn_499347inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense_31_layer_call_and_return_conditional_losses_499373inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_reshape_5_layer_call_fn_499378inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_reshape_5_layer_call_and_return_conditional_losses_499392inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_conv2d_transpose_25_layer_call_fn_499401inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_499450inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_conv2d_transpose_26_layer_call_fn_499459inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_499508inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_conv2d_transpose_27_layer_call_fn_499517inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_499566inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_conv2d_transpose_28_layer_call_fn_499575inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_499624inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
4__inference_conv2d_transpose_29_layer_call_fn_499633inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_499682inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_loss_fn_0_499702"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_1_499722"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_2_499742"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_3_499762"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_4_499782"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_loss_fn_5_499802"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
!__inference__wrapped_model_497671?%&./78@AIJ1?.
'?$
"?
input_10?????????H
? "Q?N
L
conv2d_transpose_295?2
conv2d_transpose_29??????????
O__inference_conv2d_transpose_25_layer_call_and_return_conditional_losses_499450?%&J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????H
? ?
4__inference_conv2d_transpose_25_layer_call_fn_499401?%&J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????H?
O__inference_conv2d_transpose_26_layer_call_and_return_conditional_losses_499508?./I?F
??<
:?7
inputs+???????????????????????????H
? "??<
5?2
0+???????????????????????????0
? ?
4__inference_conv2d_transpose_26_layer_call_fn_499459?./I?F
??<
:?7
inputs+???????????????????????????H
? "2?/+???????????????????????????0?
O__inference_conv2d_transpose_27_layer_call_and_return_conditional_losses_499566?78I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_conv2d_transpose_27_layer_call_fn_499517?78I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
O__inference_conv2d_transpose_28_layer_call_and_return_conditional_losses_499624?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_28_layer_call_fn_499575?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
O__inference_conv2d_transpose_29_layer_call_and_return_conditional_losses_499682?IJI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_29_layer_call_fn_499633?IJI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_dense_31_layer_call_and_return_conditional_losses_499373]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0?????????? 
? }
)__inference_dense_31_layer_call_fn_499347P/?,
%?"
 ?
inputs?????????H
? "??????????? ;
__inference_loss_fn_0_499702?

? 
? "? ;
__inference_loss_fn_1_499722%?

? 
? "? ;
__inference_loss_fn_2_499742.?

? 
? "? ;
__inference_loss_fn_3_4997627?

? 
? "? ;
__inference_loss_fn_4_499782@?

? 
? "? ;
__inference_loss_fn_5_499802I?

? 
? "? ?
C__inference_model_5_layer_call_and_return_conditional_losses_498522x%&./78@AIJ9?6
/?,
"?
input_10?????????H
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_498647x%&./78@AIJ9?6
/?,
"?
input_10?????????H
p

 
? "-?*
#? 
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_499127v%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_model_5_layer_call_and_return_conditional_losses_499338v%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p

 
? "-?*
#? 
0?????????
? ?
(__inference_model_5_layer_call_fn_498169k%&./78@AIJ9?6
/?,
"?
input_10?????????H
p 

 
? " ???????????
(__inference_model_5_layer_call_fn_498397k%&./78@AIJ9?6
/?,
"?
input_10?????????H
p

 
? " ???????????
(__inference_model_5_layer_call_fn_498887i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p 

 
? " ???????????
(__inference_model_5_layer_call_fn_498916i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p

 
? " ???????????
E__inference_reshape_5_layer_call_and_return_conditional_losses_499392b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
*__inference_reshape_5_layer_call_fn_499378U0?-
&?#
!?
inputs?????????? 
? "!????????????
$__inference_signature_wrapper_498768?%&./78@AIJ=?:
? 
3?0
.
input_10"?
input_10?????????H"Q?N
L
conv2d_transpose_295?2
conv2d_transpose_29?????????