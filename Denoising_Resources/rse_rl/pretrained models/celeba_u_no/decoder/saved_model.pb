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
conv2d_transpose_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_24/bias
?
,conv2d_transpose_24/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_24/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_24/kernel
?
.conv2d_transpose_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_24/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_23/bias
?
,conv2d_transpose_23/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_23/kernel
?
.conv2d_transpose_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_23/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_22/bias
?
,conv2d_transpose_22/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*+
shared_nameconv2d_transpose_22/kernel
?
.conv2d_transpose_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_22/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*)
shared_nameconv2d_transpose_21/bias
?
,conv2d_transpose_21/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_21/bias*
_output_shapes
:0*
dtype0
?
conv2d_transpose_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H*+
shared_nameconv2d_transpose_21/kernel
?
.conv2d_transpose_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_21/kernel*&
_output_shapes
:0H*
dtype0
?
conv2d_transpose_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*)
shared_nameconv2d_transpose_20/bias
?
,conv2d_transpose_20/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_20/bias*
_output_shapes
:H*
dtype0
?
conv2d_transpose_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*+
shared_nameconv2d_transpose_20/kernel
?
.conv2d_transpose_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_20/kernel*'
_output_shapes
:H?*
dtype0
s
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_25/bias
l
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes	
:? *
dtype0
{
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H? * 
shared_namedense_25/kernel
t
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
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
VARIABLE_VALUEdense_25/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_25/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_21/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_21/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_22/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_22/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_23/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_23/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_transpose_24/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_24/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
serving_default_input_8Placeholder*'
_output_shapes
:?????????H*
dtype0*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8dense_25/kerneldense_25/biasconv2d_transpose_20/kernelconv2d_transpose_20/biasconv2d_transpose_21/kernelconv2d_transpose_21/biasconv2d_transpose_22/kernelconv2d_transpose_22/biasconv2d_transpose_23/kernelconv2d_transpose_23/biasconv2d_transpose_24/kernelconv2d_transpose_24/bias*
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
$__inference_signature_wrapper_366197
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp.conv2d_transpose_20/kernel/Read/ReadVariableOp,conv2d_transpose_20/bias/Read/ReadVariableOp.conv2d_transpose_21/kernel/Read/ReadVariableOp,conv2d_transpose_21/bias/Read/ReadVariableOp.conv2d_transpose_22/kernel/Read/ReadVariableOp,conv2d_transpose_22/bias/Read/ReadVariableOp.conv2d_transpose_23/kernel/Read/ReadVariableOp,conv2d_transpose_23/bias/Read/ReadVariableOp.conv2d_transpose_24/kernel/Read/ReadVariableOp,conv2d_transpose_24/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_367290
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_25/kerneldense_25/biasconv2d_transpose_20/kernelconv2d_transpose_20/biasconv2d_transpose_21/kernelconv2d_transpose_21/biasconv2d_transpose_22/kernelconv2d_transpose_22/biasconv2d_transpose_23/kernelconv2d_transpose_23/biasconv2d_transpose_24/kernelconv2d_transpose_24/bias*
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
"__inference__traced_restore_367336έ
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_366556

inputs:
'dense_25_matmul_readvariableop_resource:	H? 7
(dense_25_biasadd_readvariableop_resource:	? W
<conv2d_transpose_20_conv2d_transpose_readvariableop_resource:H?A
3conv2d_transpose_20_biasadd_readvariableop_resource:HV
<conv2d_transpose_21_conv2d_transpose_readvariableop_resource:0HA
3conv2d_transpose_21_biasadd_readvariableop_resource:0V
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: 0A
3conv2d_transpose_22_biasadd_readvariableop_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_23_biasadd_readvariableop_resource:V
<conv2d_transpose_24_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_24_biasadd_readvariableop_resource:
identity??*conv2d_transpose_20/BiasAdd/ReadVariableOp?3conv2d_transpose_20/conv2d_transpose/ReadVariableOp?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_21/BiasAdd/ReadVariableOp?3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_22/BiasAdd/ReadVariableOp?3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_23/BiasAdd/ReadVariableOp?3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_24/BiasAdd/ReadVariableOp?3conv2d_transpose_24/conv2d_transpose/ReadVariableOp?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0|
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? Z
reshape_4/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_4/ReshapeReshapedense_25/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????c
conv2d_transpose_20/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_20/strided_sliceStridedSlice"conv2d_transpose_20/Shape:output:00conv2d_transpose_20/strided_slice/stack:output:02conv2d_transpose_20/strided_slice/stack_1:output:02conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose_20/stackPack*conv2d_transpose_20/strided_slice:output:0$conv2d_transpose_20/stack/1:output:0$conv2d_transpose_20/stack/2:output:0$conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_20/strided_slice_1StridedSlice"conv2d_transpose_20/stack:output:02conv2d_transpose_20/strided_slice_1/stack:output:04conv2d_transpose_20/strided_slice_1/stack_1:output:04conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
$conv2d_transpose_20/conv2d_transposeConv2DBackpropInput"conv2d_transpose_20/stack:output:0;conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
*conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose_20/BiasAddBiasAdd-conv2d_transpose_20/conv2d_transpose:output:02conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
conv2d_transpose_20/ReluRelu$conv2d_transpose_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Ho
conv2d_transpose_21/ShapeShape&conv2d_transpose_20/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_21/strided_sliceStridedSlice"conv2d_transpose_21/Shape:output:00conv2d_transpose_21/strided_slice/stack:output:02conv2d_transpose_21/strided_slice/stack_1:output:02conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_21/stackPack*conv2d_transpose_21/strided_slice:output:0$conv2d_transpose_21/stack/1:output:0$conv2d_transpose_21/stack/2:output:0$conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_21/strided_slice_1StridedSlice"conv2d_transpose_21/stack:output:02conv2d_transpose_21/strided_slice_1/stack:output:04conv2d_transpose_21/strided_slice_1/stack_1:output:04conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
$conv2d_transpose_21/conv2d_transposeConv2DBackpropInput"conv2d_transpose_21/stack:output:0;conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_20/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
*conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_21/BiasAddBiasAdd-conv2d_transpose_21/conv2d_transpose:output:02conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
conv2d_transpose_21/ReluRelu$conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0o
conv2d_transpose_22/ShapeShape&conv2d_transpose_21/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_21/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_22/ReluRelu$conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? o
conv2d_transpose_23/ShapeShape&conv2d_transpose_22/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_22/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_23/ReluRelu$conv2d_transpose_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????o
conv2d_transpose_24/ShapeShape&conv2d_transpose_23/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_24/strided_sliceStridedSlice"conv2d_transpose_24/Shape:output:00conv2d_transpose_24/strided_slice/stack:output:02conv2d_transpose_24/strided_slice/stack_1:output:02conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_24/stackPack*conv2d_transpose_24/strided_slice:output:0$conv2d_transpose_24/stack/1:output:0$conv2d_transpose_24/stack/2:output:0$conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_24/strided_slice_1StridedSlice"conv2d_transpose_24/stack:output:02conv2d_transpose_24/strided_slice_1/stack:output:04conv2d_transpose_24/strided_slice_1/stack_1:output:04conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_24/conv2d_transposeConv2DBackpropInput"conv2d_transpose_24/stack:output:0;conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_23/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_24/BiasAddBiasAdd-conv2d_transpose_24/conv2d_transpose:output:02conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_24/SigmoidSigmoid$conv2d_transpose_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: v
IdentityIdentityconv2d_transpose_24/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp+^conv2d_transpose_20/BiasAdd/ReadVariableOp4^conv2d_transpose_20/conv2d_transpose/ReadVariableOp:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_21/BiasAdd/ReadVariableOp4^conv2d_transpose_21/conv2d_transpose/ReadVariableOp:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_24/BiasAdd/ReadVariableOp4^conv2d_transpose_24/conv2d_transpose/ReadVariableOp:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_20/BiasAdd/ReadVariableOp*conv2d_transpose_20/BiasAdd/ReadVariableOp2j
3conv2d_transpose_20/conv2d_transpose/ReadVariableOp3conv2d_transpose_20/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_21/BiasAdd/ReadVariableOp*conv2d_transpose_21/BiasAdd/ReadVariableOp2j
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp3conv2d_transpose_21/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_24/BiasAdd/ReadVariableOp*conv2d_transpose_24/BiasAdd/ReadVariableOp2j
3conv2d_transpose_24/conv2d_transpose/ReadVariableOp3conv2d_transpose_24/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
ݮ
?
C__inference_model_4_layer_call_and_return_conditional_losses_365951
input_8"
dense_25_365829:	H? 
dense_25_365831:	? 5
conv2d_transpose_20_365835:H?(
conv2d_transpose_20_365837:H4
conv2d_transpose_21_365840:0H(
conv2d_transpose_21_365842:04
conv2d_transpose_22_365845: 0(
conv2d_transpose_22_365847: 4
conv2d_transpose_23_365850: (
conv2d_transpose_23_365852:4
conv2d_transpose_24_365855:(
conv2d_transpose_24_365857:
identity??+conv2d_transpose_20/StatefulPartitionedCall?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_21/StatefulPartitionedCall?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_22/StatefulPartitionedCall?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_23/StatefulPartitionedCall?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_24/StatefulPartitionedCall?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_25_365829dense_25_365831*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_365433?
reshape_4/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_20_365835conv2d_transpose_20_365837*
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0conv2d_transpose_21_365840conv2d_transpose_21_365842*
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0conv2d_transpose_22_365845conv2d_transpose_22_365847*
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0conv2d_transpose_23_365850conv2d_transpose_23_365852*
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333?
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0conv2d_transpose_24_365855conv2d_transpose_24_365857*
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_25_365829*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_365829*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_20_365835*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_20_365835*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_21_365840*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_21_365840*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_22_365845*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_22_365845*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_23_365850*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_23_365850*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_24_365855*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_24_365855*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_24/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_20/StatefulPartitionedCall:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_21/StatefulPartitionedCall:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_22/StatefulPartitionedCall:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_23/StatefulPartitionedCall:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_24/StatefulPartitionedCall:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_8
?
?
__inference_loss_fn_0_367131J
7dense_25_kernel_regularizer_abs_readvariableop_resource:	H? 
identity??.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOpf
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_25_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_25_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%dense_25/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_conv2d_transpose_22_layer_call_fn_366946

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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273?
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
??
?
C__inference_model_4_layer_call_and_return_conditional_losses_366767

inputs:
'dense_25_matmul_readvariableop_resource:	H? 7
(dense_25_biasadd_readvariableop_resource:	? W
<conv2d_transpose_20_conv2d_transpose_readvariableop_resource:H?A
3conv2d_transpose_20_biasadd_readvariableop_resource:HV
<conv2d_transpose_21_conv2d_transpose_readvariableop_resource:0HA
3conv2d_transpose_21_biasadd_readvariableop_resource:0V
<conv2d_transpose_22_conv2d_transpose_readvariableop_resource: 0A
3conv2d_transpose_22_biasadd_readvariableop_resource: V
<conv2d_transpose_23_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_23_biasadd_readvariableop_resource:V
<conv2d_transpose_24_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_24_biasadd_readvariableop_resource:
identity??*conv2d_transpose_20/BiasAdd/ReadVariableOp?3conv2d_transpose_20/conv2d_transpose/ReadVariableOp?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_21/BiasAdd/ReadVariableOp?3conv2d_transpose_21/conv2d_transpose/ReadVariableOp?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_22/BiasAdd/ReadVariableOp?3conv2d_transpose_22/conv2d_transpose/ReadVariableOp?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_23/BiasAdd/ReadVariableOp?3conv2d_transpose_23/conv2d_transpose/ReadVariableOp?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_24/BiasAdd/ReadVariableOp?3conv2d_transpose_24/conv2d_transpose/ReadVariableOp?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp?dense_25/BiasAdd/ReadVariableOp?dense_25/MatMul/ReadVariableOp?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0|
dense_25/MatMulMatMulinputs&dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? c
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? Z
reshape_4/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :\
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape_4/ReshapeReshapedense_25/Relu:activations:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????c
conv2d_transpose_20/ShapeShapereshape_4/Reshape:output:0*
T0*
_output_shapes
:q
'conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_20/strided_sliceStridedSlice"conv2d_transpose_20/Shape:output:00conv2d_transpose_20/strided_slice/stack:output:02conv2d_transpose_20/strided_slice/stack_1:output:02conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
conv2d_transpose_20/stackPack*conv2d_transpose_20/strided_slice:output:0$conv2d_transpose_20/stack/1:output:0$conv2d_transpose_20/stack/2:output:0$conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_20/strided_slice_1StridedSlice"conv2d_transpose_20/stack:output:02conv2d_transpose_20/strided_slice_1/stack:output:04conv2d_transpose_20/strided_slice_1/stack_1:output:04conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
$conv2d_transpose_20/conv2d_transposeConv2DBackpropInput"conv2d_transpose_20/stack:output:0;conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:0reshape_4/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
*conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_transpose_20/BiasAddBiasAdd-conv2d_transpose_20/conv2d_transpose:output:02conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
conv2d_transpose_20/ReluRelu$conv2d_transpose_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????Ho
conv2d_transpose_21/ShapeShape&conv2d_transpose_20/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_21/strided_sliceStridedSlice"conv2d_transpose_21/Shape:output:00conv2d_transpose_21/strided_slice/stack:output:02conv2d_transpose_21/strided_slice/stack_1:output:02conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
conv2d_transpose_21/stackPack*conv2d_transpose_21/strided_slice:output:0$conv2d_transpose_21/stack/1:output:0$conv2d_transpose_21/stack/2:output:0$conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_21/strided_slice_1StridedSlice"conv2d_transpose_21/stack:output:02conv2d_transpose_21/strided_slice_1/stack:output:04conv2d_transpose_21/strided_slice_1/stack_1:output:04conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
$conv2d_transpose_21/conv2d_transposeConv2DBackpropInput"conv2d_transpose_21/stack:output:0;conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_20/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
*conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_transpose_21/BiasAddBiasAdd-conv2d_transpose_21/conv2d_transpose:output:02conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
conv2d_transpose_21/ReluRelu$conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0o
conv2d_transpose_22/ShapeShape&conv2d_transpose_21/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_22/strided_sliceStridedSlice"conv2d_transpose_22/Shape:output:00conv2d_transpose_22/strided_slice/stack:output:02conv2d_transpose_22/strided_slice/stack_1:output:02conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_22/stackPack*conv2d_transpose_22/strided_slice:output:0$conv2d_transpose_22/stack/1:output:0$conv2d_transpose_22/stack/2:output:0$conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_22/strided_slice_1StridedSlice"conv2d_transpose_22/stack:output:02conv2d_transpose_22/strided_slice_1/stack:output:04conv2d_transpose_22/strided_slice_1/stack_1:output:04conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
$conv2d_transpose_22/conv2d_transposeConv2DBackpropInput"conv2d_transpose_22/stack:output:0;conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_21/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
*conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_22/BiasAddBiasAdd-conv2d_transpose_22/conv2d_transpose:output:02conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
conv2d_transpose_22/ReluRelu$conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? o
conv2d_transpose_23/ShapeShape&conv2d_transpose_22/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_23/strided_sliceStridedSlice"conv2d_transpose_23/Shape:output:00conv2d_transpose_23/strided_slice/stack:output:02conv2d_transpose_23/strided_slice/stack_1:output:02conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_23/stackPack*conv2d_transpose_23/strided_slice:output:0$conv2d_transpose_23/stack/1:output:0$conv2d_transpose_23/stack/2:output:0$conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_23/strided_slice_1StridedSlice"conv2d_transpose_23/stack:output:02conv2d_transpose_23/strided_slice_1/stack:output:04conv2d_transpose_23/strided_slice_1/stack_1:output:04conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_23/conv2d_transposeConv2DBackpropInput"conv2d_transpose_23/stack:output:0;conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_22/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_23/BiasAddBiasAdd-conv2d_transpose_23/conv2d_transpose:output:02conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_23/ReluRelu$conv2d_transpose_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????o
conv2d_transpose_24/ShapeShape&conv2d_transpose_23/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_24/strided_sliceStridedSlice"conv2d_transpose_24/Shape:output:00conv2d_transpose_24/strided_slice/stack:output:02conv2d_transpose_24/strided_slice/stack_1:output:02conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :]
conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_24/stackPack*conv2d_transpose_24/strided_slice:output:0$conv2d_transpose_24/stack/1:output:0$conv2d_transpose_24/stack/2:output:0$conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_24/strided_slice_1StridedSlice"conv2d_transpose_24/stack:output:02conv2d_transpose_24/strided_slice_1/stack:output:04conv2d_transpose_24/strided_slice_1/stack_1:output:04conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
$conv2d_transpose_24/conv2d_transposeConv2DBackpropInput"conv2d_transpose_24/stack:output:0;conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_23/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
*conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_24/BiasAddBiasAdd-conv2d_transpose_24/conv2d_transpose:output:02conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
conv2d_transpose_24/SigmoidSigmoid$conv2d_transpose_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: v
IdentityIdentityconv2d_transpose_24/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

NoOpNoOp+^conv2d_transpose_20/BiasAdd/ReadVariableOp4^conv2d_transpose_20/conv2d_transpose/ReadVariableOp:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_21/BiasAdd/ReadVariableOp4^conv2d_transpose_21/conv2d_transpose/ReadVariableOp:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_22/BiasAdd/ReadVariableOp4^conv2d_transpose_22/conv2d_transpose/ReadVariableOp:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_23/BiasAdd/ReadVariableOp4^conv2d_transpose_23/conv2d_transpose/ReadVariableOp:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_24/BiasAdd/ReadVariableOp4^conv2d_transpose_24/conv2d_transpose/ReadVariableOp:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_20/BiasAdd/ReadVariableOp*conv2d_transpose_20/BiasAdd/ReadVariableOp2j
3conv2d_transpose_20/conv2d_transpose/ReadVariableOp3conv2d_transpose_20/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_21/BiasAdd/ReadVariableOp*conv2d_transpose_21/BiasAdd/ReadVariableOp2j
3conv2d_transpose_21/conv2d_transpose/ReadVariableOp3conv2d_transpose_21/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_22/BiasAdd/ReadVariableOp*conv2d_transpose_22/BiasAdd/ReadVariableOp2j
3conv2d_transpose_22/conv2d_transpose/ReadVariableOp3conv2d_transpose_22/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_23/BiasAdd/ReadVariableOp*conv2d_transpose_23/BiasAdd/ReadVariableOp2j
3conv2d_transpose_23/conv2d_transpose/ReadVariableOp3conv2d_transpose_23/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_24/BiasAdd/ReadVariableOp*conv2d_transpose_24/BiasAdd/ReadVariableOp2j
3conv2d_transpose_24/conv2d_transpose/ReadVariableOp3conv2d_transpose_24/conv2d_transpose/ReadVariableOp2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_366995

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_24_layer_call_fn_367062

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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393?
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
?
?
__inference_loss_fn_1_367151]
Bconv2d_transpose_20_kernel_regularizer_abs_readvariableop_resource:H?
identity??9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_20_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_20_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_20/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp
ڮ
?
C__inference_model_4_layer_call_and_return_conditional_losses_365770

inputs"
dense_25_365648:	H? 
dense_25_365650:	? 5
conv2d_transpose_20_365654:H?(
conv2d_transpose_20_365656:H4
conv2d_transpose_21_365659:0H(
conv2d_transpose_21_365661:04
conv2d_transpose_22_365664: 0(
conv2d_transpose_22_365666: 4
conv2d_transpose_23_365669: (
conv2d_transpose_23_365671:4
conv2d_transpose_24_365674:(
conv2d_transpose_24_365676:
identity??+conv2d_transpose_20/StatefulPartitionedCall?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_21/StatefulPartitionedCall?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_22/StatefulPartitionedCall?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_23/StatefulPartitionedCall?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_24/StatefulPartitionedCall?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_365648dense_25_365650*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_365433?
reshape_4/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_20_365654conv2d_transpose_20_365656*
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0conv2d_transpose_21_365659conv2d_transpose_21_365661*
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0conv2d_transpose_22_365664conv2d_transpose_22_365666*
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0conv2d_transpose_23_365669conv2d_transpose_23_365671*
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333?
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0conv2d_transpose_24_365674conv2d_transpose_24_365676*
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_25_365648*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_365648*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_20_365654*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_20_365654*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_21_365659*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_21_365659*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_22_365664*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_22_365664*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_23_365669*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_23_365669*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_24_365674*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_24_365674*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_24/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_20/StatefulPartitionedCall:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_21/StatefulPartitionedCall:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_22/StatefulPartitionedCall:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_23/StatefulPartitionedCall:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_24/StatefulPartitionedCall:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
)__inference_dense_25_layer_call_fn_366776

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
D__inference_dense_25_layer_call_and_return_conditional_losses_365433p
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
?
?
(__inference_model_4_layer_call_fn_366316

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
C__inference_model_4_layer_call_and_return_conditional_losses_365571w
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
4__inference_conv2d_transpose_21_layer_call_fn_366888

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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213?
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
?8
?
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_366879

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_367191\
Bconv2d_transpose_22_kernel_regularizer_abs_readvariableop_resource: 0
identity??9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_22_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_22_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_22/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_5_367231\
Bconv2d_transpose_24_kernel_regularizer_abs_readvariableop_resource:
identity??9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_24_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_24_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_24/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp
?
?
4__inference_conv2d_transpose_23_layer_call_fn_367004

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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333?
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_367211\
Bconv2d_transpose_23_kernel_regularizer_abs_readvariableop_resource: 
identity??9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_23_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_23_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_23/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp
?8
?
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?4
?
"__inference__traced_restore_367336
file_prefix3
 assignvariableop_dense_25_kernel:	H? /
 assignvariableop_1_dense_25_bias:	? H
-assignvariableop_2_conv2d_transpose_20_kernel:H?9
+assignvariableop_3_conv2d_transpose_20_bias:HG
-assignvariableop_4_conv2d_transpose_21_kernel:0H9
+assignvariableop_5_conv2d_transpose_21_bias:0G
-assignvariableop_6_conv2d_transpose_22_kernel: 09
+assignvariableop_7_conv2d_transpose_22_bias: G
-assignvariableop_8_conv2d_transpose_23_kernel: 9
+assignvariableop_9_conv2d_transpose_23_bias:H
.assignvariableop_10_conv2d_transpose_24_kernel::
,assignvariableop_11_conv2d_transpose_24_bias:
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
AssignVariableOpAssignVariableOp assignvariableop_dense_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp-assignvariableop_2_conv2d_transpose_20_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_conv2d_transpose_20_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp-assignvariableop_4_conv2d_transpose_21_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp+assignvariableop_5_conv2d_transpose_21_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp-assignvariableop_6_conv2d_transpose_22_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp+assignvariableop_7_conv2d_transpose_22_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp-assignvariableop_8_conv2d_transpose_23_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_conv2d_transpose_23_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_conv2d_transpose_24_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv2d_transpose_24_biasIdentity_11:output:0"/device:CPU:0*
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
а
?
!__inference__wrapped_model_365100
input_8B
/model_4_dense_25_matmul_readvariableop_resource:	H? ?
0model_4_dense_25_biasadd_readvariableop_resource:	? _
Dmodel_4_conv2d_transpose_20_conv2d_transpose_readvariableop_resource:H?I
;model_4_conv2d_transpose_20_biasadd_readvariableop_resource:H^
Dmodel_4_conv2d_transpose_21_conv2d_transpose_readvariableop_resource:0HI
;model_4_conv2d_transpose_21_biasadd_readvariableop_resource:0^
Dmodel_4_conv2d_transpose_22_conv2d_transpose_readvariableop_resource: 0I
;model_4_conv2d_transpose_22_biasadd_readvariableop_resource: ^
Dmodel_4_conv2d_transpose_23_conv2d_transpose_readvariableop_resource: I
;model_4_conv2d_transpose_23_biasadd_readvariableop_resource:^
Dmodel_4_conv2d_transpose_24_conv2d_transpose_readvariableop_resource:I
;model_4_conv2d_transpose_24_biasadd_readvariableop_resource:
identity??2model_4/conv2d_transpose_20/BiasAdd/ReadVariableOp?;model_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOp?2model_4/conv2d_transpose_21/BiasAdd/ReadVariableOp?;model_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOp?2model_4/conv2d_transpose_22/BiasAdd/ReadVariableOp?;model_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOp?2model_4/conv2d_transpose_23/BiasAdd/ReadVariableOp?;model_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOp?2model_4/conv2d_transpose_24/BiasAdd/ReadVariableOp?;model_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOp?'model_4/dense_25/BiasAdd/ReadVariableOp?&model_4/dense_25/MatMul/ReadVariableOp?
&model_4/dense_25/MatMul/ReadVariableOpReadVariableOp/model_4_dense_25_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
model_4/dense_25/MatMulMatMulinput_8.model_4/dense_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? ?
'model_4/dense_25/BiasAdd/ReadVariableOpReadVariableOp0model_4_dense_25_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype0?
model_4/dense_25/BiasAddBiasAdd!model_4/dense_25/MatMul:product:0/model_4/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? s
model_4/dense_25/ReluRelu!model_4/dense_25/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? j
model_4/reshape_4/ShapeShape#model_4/dense_25/Relu:activations:0*
T0*
_output_shapes
:o
%model_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'model_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'model_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
model_4/reshape_4/strided_sliceStridedSlice model_4/reshape_4/Shape:output:0.model_4/reshape_4/strided_slice/stack:output:00model_4/reshape_4/strided_slice/stack_1:output:00model_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!model_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c
!model_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d
!model_4/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :??
model_4/reshape_4/Reshape/shapePack(model_4/reshape_4/strided_slice:output:0*model_4/reshape_4/Reshape/shape/1:output:0*model_4/reshape_4/Reshape/shape/2:output:0*model_4/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
model_4/reshape_4/ReshapeReshape#model_4/dense_25/Relu:activations:0(model_4/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????s
!model_4/conv2d_transpose_20/ShapeShape"model_4/reshape_4/Reshape:output:0*
T0*
_output_shapes
:y
/model_4/conv2d_transpose_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_4/conv2d_transpose_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_4/conv2d_transpose_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_4/conv2d_transpose_20/strided_sliceStridedSlice*model_4/conv2d_transpose_20/Shape:output:08model_4/conv2d_transpose_20/strided_slice/stack:output:0:model_4/conv2d_transpose_20/strided_slice/stack_1:output:0:model_4/conv2d_transpose_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_4/conv2d_transpose_20/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_20/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_20/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H?
!model_4/conv2d_transpose_20/stackPack2model_4/conv2d_transpose_20/strided_slice:output:0,model_4/conv2d_transpose_20/stack/1:output:0,model_4/conv2d_transpose_20/stack/2:output:0,model_4/conv2d_transpose_20/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_4/conv2d_transpose_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_4/conv2d_transpose_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_4/conv2d_transpose_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_4/conv2d_transpose_20/strided_slice_1StridedSlice*model_4/conv2d_transpose_20/stack:output:0:model_4/conv2d_transpose_20/strided_slice_1/stack:output:0<model_4/conv2d_transpose_20/strided_slice_1/stack_1:output:0<model_4/conv2d_transpose_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_4_conv2d_transpose_20_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype0?
,model_4/conv2d_transpose_20/conv2d_transposeConv2DBackpropInput*model_4/conv2d_transpose_20/stack:output:0Cmodel_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOp:value:0"model_4/reshape_4/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
2model_4/conv2d_transpose_20/BiasAdd/ReadVariableOpReadVariableOp;model_4_conv2d_transpose_20_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
#model_4/conv2d_transpose_20/BiasAddBiasAdd5model_4/conv2d_transpose_20/conv2d_transpose:output:0:model_4/conv2d_transpose_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H?
 model_4/conv2d_transpose_20/ReluRelu,model_4/conv2d_transpose_20/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H
!model_4/conv2d_transpose_21/ShapeShape.model_4/conv2d_transpose_20/Relu:activations:0*
T0*
_output_shapes
:y
/model_4/conv2d_transpose_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_4/conv2d_transpose_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_4/conv2d_transpose_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_4/conv2d_transpose_21/strided_sliceStridedSlice*model_4/conv2d_transpose_21/Shape:output:08model_4/conv2d_transpose_21/strided_slice/stack:output:0:model_4/conv2d_transpose_21/strided_slice/stack_1:output:0:model_4/conv2d_transpose_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_4/conv2d_transpose_21/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_21/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_21/stack/3Const*
_output_shapes
: *
dtype0*
value	B :0?
!model_4/conv2d_transpose_21/stackPack2model_4/conv2d_transpose_21/strided_slice:output:0,model_4/conv2d_transpose_21/stack/1:output:0,model_4/conv2d_transpose_21/stack/2:output:0,model_4/conv2d_transpose_21/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_4/conv2d_transpose_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_4/conv2d_transpose_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_4/conv2d_transpose_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_4/conv2d_transpose_21/strided_slice_1StridedSlice*model_4/conv2d_transpose_21/stack:output:0:model_4/conv2d_transpose_21/strided_slice_1/stack:output:0<model_4/conv2d_transpose_21/strided_slice_1/stack_1:output:0<model_4/conv2d_transpose_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_4_conv2d_transpose_21_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
,model_4/conv2d_transpose_21/conv2d_transposeConv2DBackpropInput*model_4/conv2d_transpose_21/stack:output:0Cmodel_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOp:value:0.model_4/conv2d_transpose_20/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
?
2model_4/conv2d_transpose_21/BiasAdd/ReadVariableOpReadVariableOp;model_4_conv2d_transpose_21_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
#model_4/conv2d_transpose_21/BiasAddBiasAdd5model_4/conv2d_transpose_21/conv2d_transpose:output:0:model_4/conv2d_transpose_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0?
 model_4/conv2d_transpose_21/ReluRelu,model_4/conv2d_transpose_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0
!model_4/conv2d_transpose_22/ShapeShape.model_4/conv2d_transpose_21/Relu:activations:0*
T0*
_output_shapes
:y
/model_4/conv2d_transpose_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_4/conv2d_transpose_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_4/conv2d_transpose_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_4/conv2d_transpose_22/strided_sliceStridedSlice*model_4/conv2d_transpose_22/Shape:output:08model_4/conv2d_transpose_22/strided_slice/stack:output:0:model_4/conv2d_transpose_22/strided_slice/stack_1:output:0:model_4/conv2d_transpose_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_4/conv2d_transpose_22/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_22/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_22/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
!model_4/conv2d_transpose_22/stackPack2model_4/conv2d_transpose_22/strided_slice:output:0,model_4/conv2d_transpose_22/stack/1:output:0,model_4/conv2d_transpose_22/stack/2:output:0,model_4/conv2d_transpose_22/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_4/conv2d_transpose_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_4/conv2d_transpose_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_4/conv2d_transpose_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_4/conv2d_transpose_22/strided_slice_1StridedSlice*model_4/conv2d_transpose_22/stack:output:0:model_4/conv2d_transpose_22/strided_slice_1/stack:output:0<model_4/conv2d_transpose_22/strided_slice_1/stack_1:output:0<model_4/conv2d_transpose_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_4_conv2d_transpose_22_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype0?
,model_4/conv2d_transpose_22/conv2d_transposeConv2DBackpropInput*model_4/conv2d_transpose_22/stack:output:0Cmodel_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOp:value:0.model_4/conv2d_transpose_21/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
2model_4/conv2d_transpose_22/BiasAdd/ReadVariableOpReadVariableOp;model_4_conv2d_transpose_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#model_4/conv2d_transpose_22/BiasAddBiasAdd5model_4/conv2d_transpose_22/conv2d_transpose:output:0:model_4/conv2d_transpose_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
 model_4/conv2d_transpose_22/ReluRelu,model_4/conv2d_transpose_22/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
!model_4/conv2d_transpose_23/ShapeShape.model_4/conv2d_transpose_22/Relu:activations:0*
T0*
_output_shapes
:y
/model_4/conv2d_transpose_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_4/conv2d_transpose_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_4/conv2d_transpose_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_4/conv2d_transpose_23/strided_sliceStridedSlice*model_4/conv2d_transpose_23/Shape:output:08model_4/conv2d_transpose_23/strided_slice/stack:output:0:model_4/conv2d_transpose_23/strided_slice/stack_1:output:0:model_4/conv2d_transpose_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_4/conv2d_transpose_23/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_23/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_23/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
!model_4/conv2d_transpose_23/stackPack2model_4/conv2d_transpose_23/strided_slice:output:0,model_4/conv2d_transpose_23/stack/1:output:0,model_4/conv2d_transpose_23/stack/2:output:0,model_4/conv2d_transpose_23/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_4/conv2d_transpose_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_4/conv2d_transpose_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_4/conv2d_transpose_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_4/conv2d_transpose_23/strided_slice_1StridedSlice*model_4/conv2d_transpose_23/stack:output:0:model_4/conv2d_transpose_23/strided_slice_1/stack:output:0<model_4/conv2d_transpose_23/strided_slice_1/stack_1:output:0<model_4/conv2d_transpose_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_4_conv2d_transpose_23_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,model_4/conv2d_transpose_23/conv2d_transposeConv2DBackpropInput*model_4/conv2d_transpose_23/stack:output:0Cmodel_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOp:value:0.model_4/conv2d_transpose_22/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
2model_4/conv2d_transpose_23/BiasAdd/ReadVariableOpReadVariableOp;model_4_conv2d_transpose_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#model_4/conv2d_transpose_23/BiasAddBiasAdd5model_4/conv2d_transpose_23/conv2d_transpose:output:0:model_4/conv2d_transpose_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
 model_4/conv2d_transpose_23/ReluRelu,model_4/conv2d_transpose_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
!model_4/conv2d_transpose_24/ShapeShape.model_4/conv2d_transpose_23/Relu:activations:0*
T0*
_output_shapes
:y
/model_4/conv2d_transpose_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_4/conv2d_transpose_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_4/conv2d_transpose_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_4/conv2d_transpose_24/strided_sliceStridedSlice*model_4/conv2d_transpose_24/Shape:output:08model_4/conv2d_transpose_24/strided_slice/stack:output:0:model_4/conv2d_transpose_24/strided_slice/stack_1:output:0:model_4/conv2d_transpose_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_4/conv2d_transpose_24/stack/1Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_24/stack/2Const*
_output_shapes
: *
dtype0*
value	B :e
#model_4/conv2d_transpose_24/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
!model_4/conv2d_transpose_24/stackPack2model_4/conv2d_transpose_24/strided_slice:output:0,model_4/conv2d_transpose_24/stack/1:output:0,model_4/conv2d_transpose_24/stack/2:output:0,model_4/conv2d_transpose_24/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_4/conv2d_transpose_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_4/conv2d_transpose_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_4/conv2d_transpose_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_4/conv2d_transpose_24/strided_slice_1StridedSlice*model_4/conv2d_transpose_24/stack:output:0:model_4/conv2d_transpose_24/strided_slice_1/stack:output:0<model_4/conv2d_transpose_24/strided_slice_1/stack_1:output:0<model_4/conv2d_transpose_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_4_conv2d_transpose_24_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
,model_4/conv2d_transpose_24/conv2d_transposeConv2DBackpropInput*model_4/conv2d_transpose_24/stack:output:0Cmodel_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOp:value:0.model_4/conv2d_transpose_23/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
2model_4/conv2d_transpose_24/BiasAdd/ReadVariableOpReadVariableOp;model_4_conv2d_transpose_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#model_4/conv2d_transpose_24/BiasAddBiasAdd5model_4/conv2d_transpose_24/conv2d_transpose:output:0:model_4/conv2d_transpose_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
#model_4/conv2d_transpose_24/SigmoidSigmoid,model_4/conv2d_transpose_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????~
IdentityIdentity'model_4/conv2d_transpose_24/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp3^model_4/conv2d_transpose_20/BiasAdd/ReadVariableOp<^model_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOp3^model_4/conv2d_transpose_21/BiasAdd/ReadVariableOp<^model_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOp3^model_4/conv2d_transpose_22/BiasAdd/ReadVariableOp<^model_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOp3^model_4/conv2d_transpose_23/BiasAdd/ReadVariableOp<^model_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOp3^model_4/conv2d_transpose_24/BiasAdd/ReadVariableOp<^model_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOp(^model_4/dense_25/BiasAdd/ReadVariableOp'^model_4/dense_25/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2h
2model_4/conv2d_transpose_20/BiasAdd/ReadVariableOp2model_4/conv2d_transpose_20/BiasAdd/ReadVariableOp2z
;model_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOp;model_4/conv2d_transpose_20/conv2d_transpose/ReadVariableOp2h
2model_4/conv2d_transpose_21/BiasAdd/ReadVariableOp2model_4/conv2d_transpose_21/BiasAdd/ReadVariableOp2z
;model_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOp;model_4/conv2d_transpose_21/conv2d_transpose/ReadVariableOp2h
2model_4/conv2d_transpose_22/BiasAdd/ReadVariableOp2model_4/conv2d_transpose_22/BiasAdd/ReadVariableOp2z
;model_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOp;model_4/conv2d_transpose_22/conv2d_transpose/ReadVariableOp2h
2model_4/conv2d_transpose_23/BiasAdd/ReadVariableOp2model_4/conv2d_transpose_23/BiasAdd/ReadVariableOp2z
;model_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOp;model_4/conv2d_transpose_23/conv2d_transpose/ReadVariableOp2h
2model_4/conv2d_transpose_24/BiasAdd/ReadVariableOp2model_4/conv2d_transpose_24/BiasAdd/ReadVariableOp2z
;model_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOp;model_4/conv2d_transpose_24/conv2d_transpose/ReadVariableOp2R
'model_4/dense_25/BiasAdd/ReadVariableOp'model_4/dense_25/BiasAdd/ReadVariableOp2P
&model_4/dense_25/MatMul/ReadVariableOp&model_4/dense_25/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_8
?
?
(__inference_model_4_layer_call_fn_366345

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
C__inference_model_4_layer_call_and_return_conditional_losses_365770w
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
?
?
(__inference_model_4_layer_call_fn_365598
input_8
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
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_4_layer_call_and_return_conditional_losses_365571w
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
_user_specified_name	input_8
?8
?
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_367053

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_367111

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_dense_25_layer_call_and_return_conditional_losses_365433

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOpu
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
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?8
?
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_366937

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
F
*__inference_reshape_4_layer_call_fn_366807

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
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453i
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
?
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453

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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
D__inference_dense_25_layer_call_and_return_conditional_losses_366802

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOpu
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
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_367171\
Bconv2d_transpose_21_kernel_regularizer_abs_readvariableop_resource:0H
identity??9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpq
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpBconv2d_transpose_21_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpBconv2d_transpose_21_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: n
IdentityIdentity0conv2d_transpose_21/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp
?%
?
__inference__traced_save_367290
file_prefix.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop9
5savev2_conv2d_transpose_20_kernel_read_readvariableop7
3savev2_conv2d_transpose_20_bias_read_readvariableop9
5savev2_conv2d_transpose_21_kernel_read_readvariableop7
3savev2_conv2d_transpose_21_bias_read_readvariableop9
5savev2_conv2d_transpose_22_kernel_read_readvariableop7
3savev2_conv2d_transpose_22_bias_read_readvariableop9
5savev2_conv2d_transpose_23_kernel_read_readvariableop7
3savev2_conv2d_transpose_23_bias_read_readvariableop9
5savev2_conv2d_transpose_24_kernel_read_readvariableop7
3savev2_conv2d_transpose_24_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop5savev2_conv2d_transpose_20_kernel_read_readvariableop3savev2_conv2d_transpose_20_bias_read_readvariableop5savev2_conv2d_transpose_21_kernel_read_readvariableop3savev2_conv2d_transpose_21_bias_read_readvariableop5savev2_conv2d_transpose_22_kernel_read_readvariableop3savev2_conv2d_transpose_22_bias_read_readvariableop5savev2_conv2d_transpose_23_kernel_read_readvariableop3savev2_conv2d_transpose_23_bias_read_readvariableop5savev2_conv2d_transpose_24_kernel_read_readvariableop3savev2_conv2d_transpose_24_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
(__inference_model_4_layer_call_fn_365826
input_8
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
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
C__inference_model_4_layer_call_and_return_conditional_losses_365770w
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
_user_specified_name	input_8
ݮ
?
C__inference_model_4_layer_call_and_return_conditional_losses_366076
input_8"
dense_25_365954:	H? 
dense_25_365956:	? 5
conv2d_transpose_20_365960:H?(
conv2d_transpose_20_365962:H4
conv2d_transpose_21_365965:0H(
conv2d_transpose_21_365967:04
conv2d_transpose_22_365970: 0(
conv2d_transpose_22_365972: 4
conv2d_transpose_23_365975: (
conv2d_transpose_23_365977:4
conv2d_transpose_24_365980:(
conv2d_transpose_24_365982:
identity??+conv2d_transpose_20/StatefulPartitionedCall?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_21/StatefulPartitionedCall?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_22/StatefulPartitionedCall?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_23/StatefulPartitionedCall?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_24/StatefulPartitionedCall?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinput_8dense_25_365954dense_25_365956*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_365433?
reshape_4/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_20_365960conv2d_transpose_20_365962*
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0conv2d_transpose_21_365965conv2d_transpose_21_365967*
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0conv2d_transpose_22_365970conv2d_transpose_22_365972*
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0conv2d_transpose_23_365975conv2d_transpose_23_365977*
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333?
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0conv2d_transpose_24_365980conv2d_transpose_24_365982*
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_25_365954*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_365954*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_20_365960*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_20_365960*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_21_365965*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_21_365965*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_22_365970*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_22_365970*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_23_365975*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_23_365975*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_24_365980*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_24_365980*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_24/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_20/StatefulPartitionedCall:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_21/StatefulPartitionedCall:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_22/StatefulPartitionedCall:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_23/StatefulPartitionedCall:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_24/StatefulPartitionedCall:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_8
?8
?
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp;
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
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ڮ
?
C__inference_model_4_layer_call_and_return_conditional_losses_365571

inputs"
dense_25_365434:	H? 
dense_25_365436:	? 5
conv2d_transpose_20_365455:H?(
conv2d_transpose_20_365457:H4
conv2d_transpose_21_365460:0H(
conv2d_transpose_21_365462:04
conv2d_transpose_22_365465: 0(
conv2d_transpose_22_365467: 4
conv2d_transpose_23_365470: (
conv2d_transpose_23_365472:4
conv2d_transpose_24_365475:(
conv2d_transpose_24_365477:
identity??+conv2d_transpose_20/StatefulPartitionedCall?9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_21/StatefulPartitionedCall?9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_22/StatefulPartitionedCall?9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_23/StatefulPartitionedCall?9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp?+conv2d_transpose_24/StatefulPartitionedCall?9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp?<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp? dense_25/StatefulPartitionedCall?.dense_25/kernel/Regularizer/Abs/ReadVariableOp?1dense_25/kernel/Regularizer/Square/ReadVariableOp?
 dense_25/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25_365434dense_25_365436*
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
D__inference_dense_25_layer_call_and_return_conditional_losses_365433?
reshape_4/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_365453?
+conv2d_transpose_20/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0conv2d_transpose_20_365455conv2d_transpose_20_365457*
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153?
+conv2d_transpose_21/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_20/StatefulPartitionedCall:output:0conv2d_transpose_21_365460conv2d_transpose_21_365462*
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_365213?
+conv2d_transpose_22/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_21/StatefulPartitionedCall:output:0conv2d_transpose_22_365465conv2d_transpose_22_365467*
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_365273?
+conv2d_transpose_23/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_22/StatefulPartitionedCall:output:0conv2d_transpose_23_365470conv2d_transpose_23_365472*
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_365333?
+conv2d_transpose_24/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_23/StatefulPartitionedCall:output:0conv2d_transpose_24_365475conv2d_transpose_24_365477*
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_365393f
!dense_25/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
.dense_25/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_25_365434*
_output_shapes
:	H? *
dtype0?
dense_25/kernel/Regularizer/AbsAbs6dense_25/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
dense_25/kernel/Regularizer/SumSum#dense_25/kernel/Regularizer/Abs:y:0,dense_25/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense_25/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
dense_25/kernel/Regularizer/mulMul*dense_25/kernel/Regularizer/mul/x:output:0(dense_25/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
dense_25/kernel/Regularizer/addAddV2*dense_25/kernel/Regularizer/Const:output:0#dense_25/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1dense_25/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_25_365434*
_output_shapes
:	H? *
dtype0?
"dense_25/kernel/Regularizer/SquareSquare9dense_25/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? t
#dense_25/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       ?
!dense_25/kernel/Regularizer/Sum_1Sum&dense_25/kernel/Regularizer/Square:y:0,dense_25/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#dense_25/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!dense_25/kernel/Regularizer/mul_1Mul,dense_25/kernel/Regularizer/mul_1/x:output:0*dense_25/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!dense_25/kernel/Regularizer/add_1AddV2#dense_25/kernel/Regularizer/add:z:0%dense_25/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_20_365455*'
_output_shapes
:H?*
dtype0?
*conv2d_transpose_20/kernel/Regularizer/AbsAbsAconv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_20/kernel/Regularizer/SumSum.conv2d_transpose_20/kernel/Regularizer/Abs:y:07conv2d_transpose_20/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_20/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_20/kernel/Regularizer/mulMul5conv2d_transpose_20/kernel/Regularizer/mul/x:output:03conv2d_transpose_20/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_20/kernel/Regularizer/addAddV25conv2d_transpose_20/kernel/Regularizer/Const:output:0.conv2d_transpose_20/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_20_365455*'
_output_shapes
:H?*
dtype0?
-conv2d_transpose_20/kernel/Regularizer/SquareSquareDconv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H??
.conv2d_transpose_20/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_20/kernel/Regularizer/Sum_1Sum1conv2d_transpose_20/kernel/Regularizer/Square:y:07conv2d_transpose_20/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_20/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_20/kernel/Regularizer/mul_1Mul7conv2d_transpose_20/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_20/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_20/kernel/Regularizer/add_1AddV2.conv2d_transpose_20/kernel/Regularizer/add:z:00conv2d_transpose_20/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_21_365460*&
_output_shapes
:0H*
dtype0?
*conv2d_transpose_21/kernel/Regularizer/AbsAbsAconv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_21/kernel/Regularizer/SumSum.conv2d_transpose_21/kernel/Regularizer/Abs:y:07conv2d_transpose_21/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_21/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_21/kernel/Regularizer/mulMul5conv2d_transpose_21/kernel/Regularizer/mul/x:output:03conv2d_transpose_21/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_21/kernel/Regularizer/addAddV25conv2d_transpose_21/kernel/Regularizer/Const:output:0.conv2d_transpose_21/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_21_365460*&
_output_shapes
:0H*
dtype0?
-conv2d_transpose_21/kernel/Regularizer/SquareSquareDconv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H?
.conv2d_transpose_21/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_21/kernel/Regularizer/Sum_1Sum1conv2d_transpose_21/kernel/Regularizer/Square:y:07conv2d_transpose_21/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_21/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_21/kernel/Regularizer/mul_1Mul7conv2d_transpose_21/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_21/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_21/kernel/Regularizer/add_1AddV2.conv2d_transpose_21/kernel/Regularizer/add:z:00conv2d_transpose_21/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_22_365465*&
_output_shapes
: 0*
dtype0?
*conv2d_transpose_22/kernel/Regularizer/AbsAbsAconv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_22/kernel/Regularizer/SumSum.conv2d_transpose_22/kernel/Regularizer/Abs:y:07conv2d_transpose_22/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_22/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_22/kernel/Regularizer/mulMul5conv2d_transpose_22/kernel/Regularizer/mul/x:output:03conv2d_transpose_22/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_22/kernel/Regularizer/addAddV25conv2d_transpose_22/kernel/Regularizer/Const:output:0.conv2d_transpose_22/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_22_365465*&
_output_shapes
: 0*
dtype0?
-conv2d_transpose_22/kernel/Regularizer/SquareSquareDconv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 0?
.conv2d_transpose_22/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_22/kernel/Regularizer/Sum_1Sum1conv2d_transpose_22/kernel/Regularizer/Square:y:07conv2d_transpose_22/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_22/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_22/kernel/Regularizer/mul_1Mul7conv2d_transpose_22/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_22/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_22/kernel/Regularizer/add_1AddV2.conv2d_transpose_22/kernel/Regularizer/add:z:00conv2d_transpose_22/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_23_365470*&
_output_shapes
: *
dtype0?
*conv2d_transpose_23/kernel/Regularizer/AbsAbsAconv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_23/kernel/Regularizer/SumSum.conv2d_transpose_23/kernel/Regularizer/Abs:y:07conv2d_transpose_23/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_23/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_23/kernel/Regularizer/mulMul5conv2d_transpose_23/kernel/Regularizer/mul/x:output:03conv2d_transpose_23/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_23/kernel/Regularizer/addAddV25conv2d_transpose_23/kernel/Regularizer/Const:output:0.conv2d_transpose_23/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_23_365470*&
_output_shapes
: *
dtype0?
-conv2d_transpose_23/kernel/Regularizer/SquareSquareDconv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
.conv2d_transpose_23/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_23/kernel/Regularizer/Sum_1Sum1conv2d_transpose_23/kernel/Regularizer/Square:y:07conv2d_transpose_23/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_23/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_23/kernel/Regularizer/mul_1Mul7conv2d_transpose_23/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_23/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_23/kernel/Regularizer/add_1AddV2.conv2d_transpose_23/kernel/Regularizer/add:z:00conv2d_transpose_23/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_24_365475*&
_output_shapes
:*
dtype0?
*conv2d_transpose_24/kernel/Regularizer/AbsAbsAconv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
*conv2d_transpose_24/kernel/Regularizer/SumSum.conv2d_transpose_24/kernel/Regularizer/Abs:y:07conv2d_transpose_24/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: q
,conv2d_transpose_24/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
*conv2d_transpose_24/kernel/Regularizer/mulMul5conv2d_transpose_24/kernel/Regularizer/mul/x:output:03conv2d_transpose_24/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
*conv2d_transpose_24/kernel/Regularizer/addAddV25conv2d_transpose_24/kernel/Regularizer/Const:output:0.conv2d_transpose_24/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_24_365475*&
_output_shapes
:*
dtype0?
-conv2d_transpose_24/kernel/Regularizer/SquareSquareDconv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:?
.conv2d_transpose_24/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
,conv2d_transpose_24/kernel/Regularizer/Sum_1Sum1conv2d_transpose_24/kernel/Regularizer/Square:y:07conv2d_transpose_24/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: s
.conv2d_transpose_24/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
,conv2d_transpose_24/kernel/Regularizer/mul_1Mul7conv2d_transpose_24/kernel/Regularizer/mul_1/x:output:05conv2d_transpose_24/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
,conv2d_transpose_24/kernel/Regularizer/add_1AddV2.conv2d_transpose_24/kernel/Regularizer/add:z:00conv2d_transpose_24/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: ?
IdentityIdentity4conv2d_transpose_24/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp,^conv2d_transpose_20/StatefulPartitionedCall:^conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_21/StatefulPartitionedCall:^conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_22/StatefulPartitionedCall:^conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_23/StatefulPartitionedCall:^conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp,^conv2d_transpose_24/StatefulPartitionedCall:^conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp=^conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp!^dense_25/StatefulPartitionedCall/^dense_25/kernel/Regularizer/Abs/ReadVariableOp2^dense_25/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2Z
+conv2d_transpose_20/StatefulPartitionedCall+conv2d_transpose_20/StatefulPartitionedCall2v
9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_20/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_20/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_21/StatefulPartitionedCall+conv2d_transpose_21/StatefulPartitionedCall2v
9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_21/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_21/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_22/StatefulPartitionedCall+conv2d_transpose_22/StatefulPartitionedCall2v
9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_22/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_22/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_23/StatefulPartitionedCall+conv2d_transpose_23/StatefulPartitionedCall2v
9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_23/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_23/kernel/Regularizer/Square/ReadVariableOp2Z
+conv2d_transpose_24/StatefulPartitionedCall+conv2d_transpose_24/StatefulPartitionedCall2v
9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp9conv2d_transpose_24/kernel/Regularizer/Abs/ReadVariableOp2|
<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp<conv2d_transpose_24/kernel/Regularizer/Square/ReadVariableOp2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2`
.dense_25/kernel/Regularizer/Abs/ReadVariableOp.dense_25/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_25/kernel/Regularizer/Square/ReadVariableOp1dense_25/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
4__inference_conv2d_transpose_20_layer_call_fn_366830

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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_365153?
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
?
?
$__inference_signature_wrapper_366197
input_8
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
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
!__inference__wrapped_model_365100w
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
_user_specified_name	input_8
?
a
E__inference_reshape_4_layer_call_and_return_conditional_losses_366821

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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_80
serving_default_input_8:0?????????HO
conv2d_transpose_248
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
(__inference_model_4_layer_call_fn_365598
(__inference_model_4_layer_call_fn_366316
(__inference_model_4_layer_call_fn_366345
(__inference_model_4_layer_call_fn_365826?
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
C__inference_model_4_layer_call_and_return_conditional_losses_366556
C__inference_model_4_layer_call_and_return_conditional_losses_366767
C__inference_model_4_layer_call_and_return_conditional_losses_365951
C__inference_model_4_layer_call_and_return_conditional_losses_366076?
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
!__inference__wrapped_model_365100input_8"?
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
)__inference_dense_25_layer_call_fn_366776?
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
D__inference_dense_25_layer_call_and_return_conditional_losses_366802?
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
": 	H? 2dense_25/kernel
:? 2dense_25/bias
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
*__inference_reshape_4_layer_call_fn_366807?
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_366821?
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
4__inference_conv2d_transpose_20_layer_call_fn_366830?
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_366879?
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
5:3H?2conv2d_transpose_20/kernel
&:$H2conv2d_transpose_20/bias
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
4__inference_conv2d_transpose_21_layer_call_fn_366888?
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_366937?
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
4:20H2conv2d_transpose_21/kernel
&:$02conv2d_transpose_21/bias
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
4__inference_conv2d_transpose_22_layer_call_fn_366946?
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_366995?
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
4:2 02conv2d_transpose_22/kernel
&:$ 2conv2d_transpose_22/bias
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
4__inference_conv2d_transpose_23_layer_call_fn_367004?
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_367053?
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
4:2 2conv2d_transpose_23/kernel
&:$2conv2d_transpose_23/bias
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
4__inference_conv2d_transpose_24_layer_call_fn_367062?
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_367111?
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
4:22conv2d_transpose_24/kernel
&:$2conv2d_transpose_24/bias
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
__inference_loss_fn_0_367131?
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
__inference_loss_fn_1_367151?
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
__inference_loss_fn_2_367171?
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
__inference_loss_fn_3_367191?
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
__inference_loss_fn_4_367211?
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
__inference_loss_fn_5_367231?
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
(__inference_model_4_layer_call_fn_365598input_8"?
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
(__inference_model_4_layer_call_fn_366316inputs"?
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
(__inference_model_4_layer_call_fn_366345inputs"?
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
(__inference_model_4_layer_call_fn_365826input_8"?
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
C__inference_model_4_layer_call_and_return_conditional_losses_366556inputs"?
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
C__inference_model_4_layer_call_and_return_conditional_losses_366767inputs"?
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
C__inference_model_4_layer_call_and_return_conditional_losses_365951input_8"?
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
C__inference_model_4_layer_call_and_return_conditional_losses_366076input_8"?
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
$__inference_signature_wrapper_366197input_8"?
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
)__inference_dense_25_layer_call_fn_366776inputs"?
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
D__inference_dense_25_layer_call_and_return_conditional_losses_366802inputs"?
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
*__inference_reshape_4_layer_call_fn_366807inputs"?
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
E__inference_reshape_4_layer_call_and_return_conditional_losses_366821inputs"?
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
4__inference_conv2d_transpose_20_layer_call_fn_366830inputs"?
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
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_366879inputs"?
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
4__inference_conv2d_transpose_21_layer_call_fn_366888inputs"?
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
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_366937inputs"?
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
4__inference_conv2d_transpose_22_layer_call_fn_366946inputs"?
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
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_366995inputs"?
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
4__inference_conv2d_transpose_23_layer_call_fn_367004inputs"?
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
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_367053inputs"?
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
4__inference_conv2d_transpose_24_layer_call_fn_367062inputs"?
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
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_367111inputs"?
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
__inference_loss_fn_0_367131"?
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
__inference_loss_fn_1_367151"?
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
__inference_loss_fn_2_367171"?
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
__inference_loss_fn_3_367191"?
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
__inference_loss_fn_4_367211"?
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
__inference_loss_fn_5_367231"?
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
!__inference__wrapped_model_365100?%&./78@AIJ0?-
&?#
!?
input_8?????????H
? "Q?N
L
conv2d_transpose_245?2
conv2d_transpose_24??????????
O__inference_conv2d_transpose_20_layer_call_and_return_conditional_losses_366879?%&J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????H
? ?
4__inference_conv2d_transpose_20_layer_call_fn_366830?%&J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????H?
O__inference_conv2d_transpose_21_layer_call_and_return_conditional_losses_366937?./I?F
??<
:?7
inputs+???????????????????????????H
? "??<
5?2
0+???????????????????????????0
? ?
4__inference_conv2d_transpose_21_layer_call_fn_366888?./I?F
??<
:?7
inputs+???????????????????????????H
? "2?/+???????????????????????????0?
O__inference_conv2d_transpose_22_layer_call_and_return_conditional_losses_366995?78I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
4__inference_conv2d_transpose_22_layer_call_fn_366946?78I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
O__inference_conv2d_transpose_23_layer_call_and_return_conditional_losses_367053?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_23_layer_call_fn_367004?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
O__inference_conv2d_transpose_24_layer_call_and_return_conditional_losses_367111?IJI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
4__inference_conv2d_transpose_24_layer_call_fn_367062?IJI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
D__inference_dense_25_layer_call_and_return_conditional_losses_366802]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0?????????? 
? }
)__inference_dense_25_layer_call_fn_366776P/?,
%?"
 ?
inputs?????????H
? "??????????? ;
__inference_loss_fn_0_367131?

? 
? "? ;
__inference_loss_fn_1_367151%?

? 
? "? ;
__inference_loss_fn_2_367171.?

? 
? "? ;
__inference_loss_fn_3_3671917?

? 
? "? ;
__inference_loss_fn_4_367211@?

? 
? "? ;
__inference_loss_fn_5_367231I?

? 
? "? ?
C__inference_model_4_layer_call_and_return_conditional_losses_365951w%&./78@AIJ8?5
.?+
!?
input_8?????????H
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_366076w%&./78@AIJ8?5
.?+
!?
input_8?????????H
p

 
? "-?*
#? 
0?????????
? ?
C__inference_model_4_layer_call_and_return_conditional_losses_366556v%&./78@AIJ7?4
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
C__inference_model_4_layer_call_and_return_conditional_losses_366767v%&./78@AIJ7?4
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
(__inference_model_4_layer_call_fn_365598j%&./78@AIJ8?5
.?+
!?
input_8?????????H
p 

 
? " ???????????
(__inference_model_4_layer_call_fn_365826j%&./78@AIJ8?5
.?+
!?
input_8?????????H
p

 
? " ???????????
(__inference_model_4_layer_call_fn_366316i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p 

 
? " ???????????
(__inference_model_4_layer_call_fn_366345i%&./78@AIJ7?4
-?*
 ?
inputs?????????H
p

 
? " ???????????
E__inference_reshape_4_layer_call_and_return_conditional_losses_366821b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
*__inference_reshape_4_layer_call_fn_366807U0?-
&?#
!?
inputs?????????? 
? "!????????????
$__inference_signature_wrapper_366197?%&./78@AIJ;?8
? 
1?.
,
input_8!?
input_8?????????H"Q?N
L
conv2d_transpose_245?2
conv2d_transpose_24?????????