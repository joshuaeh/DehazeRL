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
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0**
shared_nameconv2d_transpose_2/kernel
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:0*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:0H*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:H*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*'
_output_shapes
:H?*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:? *
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H? *
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
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
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ga
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEconv2d_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEconv2d_transpose_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEconv2d_transpose_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????H*
dtype0*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_5/kerneldense_5/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/bias*
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
$__inference_signature_wrapper_215741
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_216834
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/bias*
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
"__inference__traced_restore_216880݅
?
D
(__inference_reshape_layer_call_fn_216351

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
GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_214997i
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
?8
?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_216539

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;
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
-:+??????????????????????????? p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_layer_call_fn_216374

inputs"
unknown:H?
	unknown_0:H
identity??StatefulPartitionedCall?
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
GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697?
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????0p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
ū
?
A__inference_model_layer_call_and_return_conditional_losses_215495
input_2!
dense_5_215373:	H? 
dense_5_215375:	? 2
conv2d_transpose_215379:H?%
conv2d_transpose_215381:H3
conv2d_transpose_1_215384:0H'
conv2d_transpose_1_215386:03
conv2d_transpose_2_215389: 0'
conv2d_transpose_2_215391: 3
conv2d_transpose_3_215394: '
conv2d_transpose_3_215396:3
conv2d_transpose_4_215399:'
conv2d_transpose_4_215401:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_215373dense_5_215375*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_214977?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_214997?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_215379conv2d_transpose_215381*
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
GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_215384conv2d_transpose_1_215386*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_215389conv2d_transpose_2_215391*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_215394conv2d_transpose_3_215396*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_215399conv2d_transpose_4_215401*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_215373*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_215373*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_215379*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_215379*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_215384*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_215384*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_215389*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_215389*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_215394*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_215394*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_215399*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_215399*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?8
?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_216655

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_4_layer_call_fn_216606

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937?
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
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_216365

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
?7
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????Hn
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?8
?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;
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
-:+??????????????????????????? p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?7
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_216423

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????Hn
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_216755[
Aconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource: 
identity??8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpp
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: m
IdentityIdentity/conv2d_transpose_3/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp
?8
?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?8
?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_216481

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????0p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_1_layer_call_fn_216432

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757?
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
?
&__inference_model_layer_call_fn_215860

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
GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_215115w
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
3__inference_conv2d_transpose_2_layer_call_fn_216490

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817?
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
?
?
__inference_loss_fn_3_216735[
Aconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource: 0
identity??8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpp
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: m
IdentityIdentity/conv2d_transpose_2/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp
ū
?
A__inference_model_layer_call_and_return_conditional_losses_215620
input_2!
dense_5_215498:	H? 
dense_5_215500:	? 2
conv2d_transpose_215504:H?%
conv2d_transpose_215506:H3
conv2d_transpose_1_215509:0H'
conv2d_transpose_1_215511:03
conv2d_transpose_2_215514: 0'
conv2d_transpose_2_215516: 3
conv2d_transpose_3_215519: '
conv2d_transpose_3_215521:3
conv2d_transpose_4_215524:'
conv2d_transpose_4_215526:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_5_215498dense_5_215500*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_214977?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_214997?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_215504conv2d_transpose_215506*
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
GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_215509conv2d_transpose_1_215511*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_215514conv2d_transpose_2_215516*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_215519conv2d_transpose_3_215521*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_215524conv2d_transpose_4_215526*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_215498*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_215498*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_215504*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_215504*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_215509*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_215509*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_215514*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_215514*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_215519*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_215519*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_215524*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_215524*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?
?
$__inference_signature_wrapper_215741
input_2
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_214644w
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
«
?
A__inference_model_layer_call_and_return_conditional_losses_215314

inputs!
dense_5_215192:	H? 
dense_5_215194:	? 2
conv2d_transpose_215198:H?%
conv2d_transpose_215200:H3
conv2d_transpose_1_215203:0H'
conv2d_transpose_1_215205:03
conv2d_transpose_2_215208: 0'
conv2d_transpose_2_215210: 3
conv2d_transpose_3_215213: '
conv2d_transpose_3_215215:3
conv2d_transpose_4_215218:'
conv2d_transpose_4_215220:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_215192dense_5_215194*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_214977?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_214997?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_215198conv2d_transpose_215200*
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
GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_215203conv2d_transpose_1_215205*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_215208conv2d_transpose_2_215210*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_215213conv2d_transpose_3_215215*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_215218conv2d_transpose_4_215220*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_215192*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_215192*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_215198*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_215198*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_215203*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_215203*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_215208*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_215208*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_215213*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_215213*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_215218*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_215218*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_215889

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
GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_215314w
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
??
?
A__inference_model_layer_call_and_return_conditional_losses_216100

inputs9
&dense_5_matmul_readvariableop_resource:	H? 6
'dense_5_biasadd_readvariableop_resource:	? T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:H?>
0conv2d_transpose_biasadd_readvariableop_resource:HU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_1_biasadd_readvariableop_resource:0U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? W
reshape/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_5/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Hk
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_4/SigmoidSigmoid#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: u
IdentityIdentityconv2d_transpose_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_216775[
Aconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource:
identity??8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpp
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: m
IdentityIdentity/conv2d_transpose_4/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_0_216675I
6dense_5_kernel_regularizer_abs_readvariableop_resource:	H? 
identity??-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOpe
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentity$dense_5/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_2_216715[
Aconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource:0H
identity??8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpp
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: m
IdentityIdentity/conv2d_transpose_1/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp
?
?
&__inference_model_layer_call_fn_215370
input_2
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_215314w
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
«
?
A__inference_model_layer_call_and_return_conditional_losses_215115

inputs!
dense_5_214978:	H? 
dense_5_214980:	? 2
conv2d_transpose_214999:H?%
conv2d_transpose_215001:H3
conv2d_transpose_1_215004:0H'
conv2d_transpose_1_215006:03
conv2d_transpose_2_215009: 0'
conv2d_transpose_2_215011: 3
conv2d_transpose_3_215014: '
conv2d_transpose_3_215016:3
conv2d_transpose_4_215019:'
conv2d_transpose_4_215021:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_214978dense_5_214980*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_214977?
reshape/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_214997?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_214999conv2d_transpose_215001*
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
GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_214697?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_215004conv2d_transpose_1_215006*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_214757?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_215009conv2d_transpose_2_215011*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_214817?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_215014conv2d_transpose_3_215016*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_215019conv2d_transpose_4_215021*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_214937e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_214978*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_214978*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_214999*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_214999*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_215004*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_215004*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_215009*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_215009*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_215014*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_215014*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_215019*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_215019*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_214997

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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_3_layer_call_fn_216548

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_214877?
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
?8
?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_216597

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;
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
-:+???????????????????????????p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_215142
input_2
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_215115w
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
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?
?
__inference_loss_fn_1_216695Z
?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource:H?
identity??6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpn
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: k
IdentityIdentity-conv2d_transpose/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp
?%
?
__inference__traced_save_216834
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
??
?
!__inference__wrapped_model_214644
input_2?
,model_dense_5_matmul_readvariableop_resource:	H? <
-model_dense_5_biasadd_readvariableop_resource:	? Z
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:H?D
6model_conv2d_transpose_biasadd_readvariableop_resource:H[
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0HF
8model_conv2d_transpose_1_biasadd_readvariableop_resource:0[
Amodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0F
8model_conv2d_transpose_2_biasadd_readvariableop_resource: [
Amodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: F
8model_conv2d_transpose_3_biasadd_readvariableop_resource:[
Amodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:F
8model_conv2d_transpose_4_biasadd_readvariableop_resource:
identity??-model/conv2d_transpose/BiasAdd/ReadVariableOp?6model/conv2d_transpose/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_1/BiasAdd/ReadVariableOp?8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_2/BiasAdd/ReadVariableOp?8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_3/BiasAdd/ReadVariableOp?8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_4/BiasAdd/ReadVariableOp?8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?$model/dense_5/BiasAdd/ReadVariableOp?#model/dense_5/MatMul/ReadVariableOp?
#model/dense_5/MatMul/ReadVariableOpReadVariableOp,model_dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
model/dense_5/MatMulMatMulinput_2+model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
$model/dense_5/BiasAdd/ReadVariableOpReadVariableOp-model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
model/dense_5/BiasAddBiasAddmodel/dense_5/MatMul:product:0,model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? m
model/dense_5/ReluRelumodel/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? c
model/reshape/ShapeShape model/dense_5/Relu:activations:0*
T0*
_output_shapes
:k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :_
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model/reshape/ReshapeReshape model/dense_5/Relu:activations:0$model/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????j
model/conv2d_transpose/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
:t
*model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$model/conv2d_transpose/strided_sliceStridedSlice%model/conv2d_transpose/Shape:output:03model/conv2d_transpose/strided_slice/stack:output:05model/conv2d_transpose/strided_slice/stack_1:output:05model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
model/conv2d_transpose/stackPack-model/conv2d_transpose/strided_slice:output:0'model/conv2d_transpose/stack/1:output:0'model/conv2d_transpose/stack/2:output:0'model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:v
,model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:0>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0model/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:05model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
model/conv2d_transpose/ReluRelu'model/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Hw
model/conv2d_transpose_1/ShapeShape)model/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv2d_transpose_1/strided_sliceStridedSlice'model/conv2d_transpose_1/Shape:output:05model/conv2d_transpose_1/strided_slice/stack:output:07model/conv2d_transpose_1/strided_slice/stack_1:output:07model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
model/conv2d_transpose_1/stackPack/model/conv2d_transpose_1/strided_slice:output:0)model/conv2d_transpose_1/stack/1:output:0)model/conv2d_transpose_1/stack/2:output:0)model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:0@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0)model/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:07model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
model/conv2d_transpose_1/ReluRelu)model/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0y
model/conv2d_transpose_2/ShapeShape+model/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv2d_transpose_2/strided_sliceStridedSlice'model/conv2d_transpose_2/Shape:output:05model/conv2d_transpose_2/strided_slice/stack:output:07model/conv2d_transpose_2/strided_slice/stack_1:output:07model/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
model/conv2d_transpose_2/stackPack/model/conv2d_transpose_2/strided_slice:output:0)model/conv2d_transpose_2/stack/1:output:0)model/conv2d_transpose_2/stack/2:output:0)model/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model/conv2d_transpose_2/strided_slice_1StridedSlice'model/conv2d_transpose_2/stack:output:07model/conv2d_transpose_2/strided_slice_1/stack:output:09model/conv2d_transpose_2/strided_slice_1/stack_1:output:09model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_2/stack:output:0@model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
/model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 model/conv2d_transpose_2/BiasAddBiasAdd2model/conv2d_transpose_2/conv2d_transpose:output:07model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
model/conv2d_transpose_2/ReluRelu)model/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? y
model/conv2d_transpose_3/ShapeShape+model/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv2d_transpose_3/strided_sliceStridedSlice'model/conv2d_transpose_3/Shape:output:05model/conv2d_transpose_3/strided_slice/stack:output:07model/conv2d_transpose_3/strided_slice/stack_1:output:07model/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv2d_transpose_3/stackPack/model/conv2d_transpose_3/strided_slice:output:0)model/conv2d_transpose_3/stack/1:output:0)model/conv2d_transpose_3/stack/2:output:0)model/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model/conv2d_transpose_3/strided_slice_1StridedSlice'model/conv2d_transpose_3/stack:output:07model/conv2d_transpose_3/strided_slice_1/stack:output:09model/conv2d_transpose_3/strided_slice_1/stack_1:output:09model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
)model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_3/stack:output:0@model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
/model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 model/conv2d_transpose_3/BiasAddBiasAdd2model/conv2d_transpose_3/conv2d_transpose:output:07model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
model/conv2d_transpose_3/ReluRelu)model/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????y
model/conv2d_transpose_4/ShapeShape+model/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:v
,model/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&model/conv2d_transpose_4/strided_sliceStridedSlice'model/conv2d_transpose_4/Shape:output:05model/conv2d_transpose_4/strided_slice/stack:output:07model/conv2d_transpose_4/strided_slice/stack_1:output:07model/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 model/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 model/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
model/conv2d_transpose_4/stackPack/model/conv2d_transpose_4/strided_slice:output:0)model/conv2d_transpose_4/stack/1:output:0)model/conv2d_transpose_4/stack/2:output:0)model/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model/conv2d_transpose_4/strided_slice_1StridedSlice'model/conv2d_transpose_4/stack:output:07model/conv2d_transpose_4/strided_slice_1/stack:output:09model/conv2d_transpose_4/strided_slice_1/stack_1:output:09model/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
)model/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_4/stack:output:0@model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 model/conv2d_transpose_4/BiasAddBiasAdd2model/conv2d_transpose_4/conv2d_transpose:output:07model/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model/conv2d_transpose_4/SigmoidSigmoid)model/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????{
IdentityIdentity$model/conv2d_transpose_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_2/BiasAdd/ReadVariableOp9^model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_3/BiasAdd/ReadVariableOp9^model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_4/BiasAdd/ReadVariableOp9^model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp%^model/dense_5/BiasAdd/ReadVariableOp$^model/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2^
-model/conv2d_transpose/BiasAdd/ReadVariableOp-model/conv2d_transpose/BiasAdd/ReadVariableOp2p
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp6model/conv2d_transpose/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp/model/conv2d_transpose_1/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp/model/conv2d_transpose_2/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp/model/conv2d_transpose_3/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_4/BiasAdd/ReadVariableOp/model/conv2d_transpose_4/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2L
$model/dense_5/BiasAdd/ReadVariableOp$model/dense_5/BiasAdd/ReadVariableOp2J
#model/dense_5/MatMul/ReadVariableOp#model/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_214977

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOpu
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
:?????????? e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_216311

inputs9
&dense_5_matmul_readvariableop_resource:	H? 6
'dense_5_biasadd_readvariableop_resource:	? T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:H?>
0conv2d_transpose_biasadd_readvariableop_resource:HU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_1_biasadd_readvariableop_resource:0U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0z
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? a
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? W
reshape/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_5/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hz
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Hk
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0~
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0m
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ~
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? m
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_4/SigmoidSigmoid#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: n
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: p
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: p
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: r
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: u
IdentityIdentityconv2d_transpose_4/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
C__inference_dense_5_layer_call_and_return_conditional_losses_216346

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOpu
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
:?????????? e
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: e
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? s
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: g
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
(__inference_dense_5_layer_call_fn_216320

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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_214977p
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
?4
?
"__inference__traced_restore_216880
file_prefix2
assignvariableop_dense_5_kernel:	H? .
assignvariableop_1_dense_5_bias:	? E
*assignvariableop_2_conv2d_transpose_kernel:H?6
(assignvariableop_3_conv2d_transpose_bias:HF
,assignvariableop_4_conv2d_transpose_1_kernel:0H8
*assignvariableop_5_conv2d_transpose_1_bias:0F
,assignvariableop_6_conv2d_transpose_2_kernel: 08
*assignvariableop_7_conv2d_transpose_2_bias: F
,assignvariableop_8_conv2d_transpose_3_kernel: 8
*assignvariableop_9_conv2d_transpose_3_bias:G
-assignvariableop_10_conv2d_transpose_4_kernel:9
+assignvariableop_11_conv2d_transpose_4_bias:
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
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv2d_transpose_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv2d_transpose_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2d_transpose_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_transpose_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_4_biasIdentity_11:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????HN
conv2d_transpose_48
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
&__inference_model_layer_call_fn_215142
&__inference_model_layer_call_fn_215860
&__inference_model_layer_call_fn_215889
&__inference_model_layer_call_fn_215370?
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
A__inference_model_layer_call_and_return_conditional_losses_216100
A__inference_model_layer_call_and_return_conditional_losses_216311
A__inference_model_layer_call_and_return_conditional_losses_215495
A__inference_model_layer_call_and_return_conditional_losses_215620?
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
!__inference__wrapped_model_214644input_2"?
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
(__inference_dense_5_layer_call_fn_216320?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_216346?
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
!:	H? 2dense_5/kernel
:? 2dense_5/bias
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
(__inference_reshape_layer_call_fn_216351?
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
C__inference_reshape_layer_call_and_return_conditional_losses_216365?
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
1__inference_conv2d_transpose_layer_call_fn_216374?
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
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_216423?
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
2:0H?2conv2d_transpose/kernel
#:!H2conv2d_transpose/bias
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
3__inference_conv2d_transpose_1_layer_call_fn_216432?
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
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_216481?
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
3:10H2conv2d_transpose_1/kernel
%:#02conv2d_transpose_1/bias
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
3__inference_conv2d_transpose_2_layer_call_fn_216490?
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
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_216539?
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
3:1 02conv2d_transpose_2/kernel
%:# 2conv2d_transpose_2/bias
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
3__inference_conv2d_transpose_3_layer_call_fn_216548?
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
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_216597?
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
3:1 2conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
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
3__inference_conv2d_transpose_4_layer_call_fn_216606?
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
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_216655?
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
3:12conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
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
__inference_loss_fn_0_216675?
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
__inference_loss_fn_1_216695?
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
__inference_loss_fn_2_216715?
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
__inference_loss_fn_3_216735?
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
__inference_loss_fn_4_216755?
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
__inference_loss_fn_5_216775?
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
&__inference_model_layer_call_fn_215142input_2"?
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
&__inference_model_layer_call_fn_215860inputs"?
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
&__inference_model_layer_call_fn_215889inputs"?
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
&__inference_model_layer_call_fn_215370input_2"?
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
A__inference_model_layer_call_and_return_conditional_losses_216100inputs"?
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
A__inference_model_layer_call_and_return_conditional_losses_216311inputs"?
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
A__inference_model_layer_call_and_return_conditional_losses_215495input_2"?
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
A__inference_model_layer_call_and_return_conditional_losses_215620input_2"?
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
$__inference_signature_wrapper_215741input_2"?
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
(__inference_dense_5_layer_call_fn_216320inputs"?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_216346inputs"?
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
(__inference_reshape_layer_call_fn_216351inputs"?
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
C__inference_reshape_layer_call_and_return_conditional_losses_216365inputs"?
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
1__inference_conv2d_transpose_layer_call_fn_216374inputs"?
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
?B?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_216423inputs"?
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
3__inference_conv2d_transpose_1_layer_call_fn_216432inputs"?
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
?B?
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_216481inputs"?
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
3__inference_conv2d_transpose_2_layer_call_fn_216490inputs"?
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
?B?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_216539inputs"?
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
3__inference_conv2d_transpose_3_layer_call_fn_216548inputs"?
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
?B?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_216597inputs"?
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
3__inference_conv2d_transpose_4_layer_call_fn_216606inputs"?
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
?B?
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_216655inputs"?
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
__inference_loss_fn_0_216675"?
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
__inference_loss_fn_1_216695"?
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
__inference_loss_fn_2_216715"?
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
__inference_loss_fn_3_216735"?
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
__inference_loss_fn_4_216755"?
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
__inference_loss_fn_5_216775"?
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
!__inference__wrapped_model_214644?%&./78@AIJ0?-
&?#
!?
input_2?????????H
? "O?L
J
conv2d_transpose_44?1
conv2d_transpose_4??????????
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_216481?./I?F
??<
:?7
inputs+???????????????????????????H
? "??<
5?2
0+???????????????????????????0
? ?
3__inference_conv2d_transpose_1_layer_call_fn_216432?./I?F
??<
:?7
inputs+???????????????????????????H
? "2?/+???????????????????????????0?
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_216539?78I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_2_layer_call_fn_216490?78I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_216597?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_3_layer_call_fn_216548?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_216655?IJI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_4_layer_call_fn_216606?IJI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_216423?%&J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????H
? ?
1__inference_conv2d_transpose_layer_call_fn_216374?%&J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????H?
C__inference_dense_5_layer_call_and_return_conditional_losses_216346]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0?????????? 
? |
(__inference_dense_5_layer_call_fn_216320P/?,
%?"
 ?
inputs?????????H
? "??????????? ;
__inference_loss_fn_0_216675?

? 
? "? ;
__inference_loss_fn_1_216695%?

? 
? "? ;
__inference_loss_fn_2_216715.?

? 
? "? ;
__inference_loss_fn_3_2167357?

? 
? "? ;
__inference_loss_fn_4_216755@?

? 
? "? ;
__inference_loss_fn_5_216775I?

? 
? "? ?
A__inference_model_layer_call_and_return_conditional_losses_215495w%&./78@AIJ8?5
.?+
!?
input_2?????????H
p 

 
? "-?*
#? 
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_215620w%&./78@AIJ8?5
.?+
!?
input_2?????????H
p

 
? "-?*
#? 
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_216100v%&./78@AIJ7?4
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
A__inference_model_layer_call_and_return_conditional_losses_216311v%&./78@AIJ7?4
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
&__inference_model_layer_call_fn_215142j%&./78@AIJ8?5
.?+
!?
input_2?????????H
p 

 
? " ???????????
&__inference_model_layer_call_fn_215370j%&./78@AIJ8?5
.?+
!?
input_2?????????H
p

 
? " ???????????
&__inference_model_layer_call_fn_215860i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p 

 
? " ???????????
&__inference_model_layer_call_fn_215889i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p

 
? " ???????????
C__inference_reshape_layer_call_and_return_conditional_losses_216365b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
(__inference_reshape_layer_call_fn_216351U0?-
&?#
!?
inputs?????????? 
? "!????????????
$__inference_signature_wrapper_215741?%&./78@AIJ;?8
? 
1?.
,
input_2!?
input_2?????????H"O?L
J
conv2d_transpose_44?1
conv2d_transpose_4?????????