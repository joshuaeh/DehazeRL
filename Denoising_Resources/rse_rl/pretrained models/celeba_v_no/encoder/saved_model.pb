ª
Ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Conv2D

input"T
filter"T
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
û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38è
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:H*
dtype0
{
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H* 
shared_namedense_27/kernel
t
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes
:	H*
dtype0
d
z/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namez/bias
]
z/bias/Read/ReadVariableOpReadVariableOpz/bias*
_output_shapes
:H*
dtype0
m
z/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H*
shared_name
z/kernel
f
z/kernel/Read/ReadVariableOpReadVariableOpz/kernel*
_output_shapes
:	H*
dtype0
s
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
l
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes	
:*
dtype0
|
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
	* 
shared_namedense_26/kernel
u
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel* 
_output_shapes
:
	*
dtype0
¢
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:H*
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:H*
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:H*
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:H*
dtype0
t
conv2d_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_15/bias
m
"conv2d_15/bias/Read/ReadVariableOpReadVariableOpconv2d_15/bias*
_output_shapes
:H*
dtype0

conv2d_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H*!
shared_nameconv2d_15/kernel
}
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*&
_output_shapes
:@H*
dtype0
t
conv2d_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_14/bias
m
"conv2d_14/bias/Read/ReadVariableOpReadVariableOpconv2d_14/bias*
_output_shapes
:@*
dtype0

conv2d_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_14/kernel
}
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0

conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
: *
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:*
dtype0

conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
çC
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¢C
valueCBC BC

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
È
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
È
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
È
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
È
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
Õ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*

B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
¦
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*
¦
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
¦
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias*

0
1
"2
#3
+4
,5
46
57
>8
?9
@10
A11
N12
O13
V14
W15
^16
_17*
z
0
1
"2
#3
+4
,5
46
57
>8
?9
N10
O11
V12
W13
^14
_15*

`0
a1
b2
c3* 
°
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
6
mtrace_0
ntrace_1
otrace_2
ptrace_3* 
* 

qserving_default* 

0
1*

0
1*
	
`0* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

wtrace_0* 

xtrace_0* 
`Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1*

"0
#1*
	
a0* 

ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

+0
,1*

+0
,1*
	
b0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_14/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_14/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

40
51*

40
51*
	
c0* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_15/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_15/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

N0
O1*

N0
O1*
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

£trace_0* 

¤trace_0* 
_Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

ªtrace_0* 

«trace_0* 
XR
VARIABLE_VALUEz/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEz/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

^0
_1*
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

±trace_0* 

²trace_0* 
_Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_27/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

³trace_0* 

´trace_0* 

µtrace_0* 

¶trace_0* 

@0
A1*
J
0
1
2
3
4
5
6
7
	8

9*
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
	
`0* 
* 
* 
* 
* 
* 
* 
	
a0* 
* 
* 
* 
* 
* 
* 
	
b0* 
* 
* 
* 
* 
* 
* 
	
c0* 
* 
* 
* 

@0
A1*
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

serving_default_input_9Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ê
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/biasz/kernelz/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_496564
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
õ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp"conv2d_14/bias/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp"conv2d_15/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_497398

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasconv2d_14/kernelconv2d_14/biasconv2d_15/kernelconv2d_15/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancedense_26/kerneldense_26/biasz/kernelz/biasdense_27/kerneldense_27/bias*
Tin
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_497462¤

å
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

å
E__inference_conv2d_15_layer_call_and_return_conditional_losses_497108

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHg
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHÞ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
º

"__inference_z_layer_call_fn_497210

inputs
unknown:	H
	unknown_0:H
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_495815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

)__inference_dense_27_layer_call_fn_497229

inputs
unknown:	H
	unknown_0:H
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_495799o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
F
*__inference_flatten_3_layer_call_fn_497175

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Ô
ê
__inference_loss_fn_2_497300R
8conv2d_14_kernel_regularizer_abs_readvariableop_resource: @
identity¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_14_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ³
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_14_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_14/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp

À
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495608

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿHÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Î^
à
!__inference__wrapped_model_495555
input_9J
0encoder_conv2d_12_conv2d_readvariableop_resource:?
1encoder_conv2d_12_biasadd_readvariableop_resource:J
0encoder_conv2d_13_conv2d_readvariableop_resource: ?
1encoder_conv2d_13_biasadd_readvariableop_resource: J
0encoder_conv2d_14_conv2d_readvariableop_resource: @?
1encoder_conv2d_14_biasadd_readvariableop_resource:@J
0encoder_conv2d_15_conv2d_readvariableop_resource:@H?
1encoder_conv2d_15_biasadd_readvariableop_resource:HC
5encoder_batch_normalization_3_readvariableop_resource:HE
7encoder_batch_normalization_3_readvariableop_1_resource:HT
Fencoder_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:HV
Hencoder_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:HC
/encoder_dense_26_matmul_readvariableop_resource:
	?
0encoder_dense_26_biasadd_readvariableop_resource:	B
/encoder_dense_27_matmul_readvariableop_resource:	H>
0encoder_dense_27_biasadd_readvariableop_resource:H;
(encoder_z_matmul_readvariableop_resource:	H7
)encoder_z_biasadd_readvariableop_resource:H
identity

identity_1¢=encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢?encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢,encoder/batch_normalization_3/ReadVariableOp¢.encoder/batch_normalization_3/ReadVariableOp_1¢(encoder/conv2d_12/BiasAdd/ReadVariableOp¢'encoder/conv2d_12/Conv2D/ReadVariableOp¢(encoder/conv2d_13/BiasAdd/ReadVariableOp¢'encoder/conv2d_13/Conv2D/ReadVariableOp¢(encoder/conv2d_14/BiasAdd/ReadVariableOp¢'encoder/conv2d_14/Conv2D/ReadVariableOp¢(encoder/conv2d_15/BiasAdd/ReadVariableOp¢'encoder/conv2d_15/Conv2D/ReadVariableOp¢'encoder/dense_26/BiasAdd/ReadVariableOp¢&encoder/dense_26/MatMul/ReadVariableOp¢'encoder/dense_27/BiasAdd/ReadVariableOp¢&encoder/dense_27/MatMul/ReadVariableOp¢ encoder/z/BiasAdd/ReadVariableOp¢encoder/z/MatMul/ReadVariableOp 
'encoder/conv2d_12/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¾
encoder/conv2d_12/Conv2DConv2Dinput_9/encoder/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

(encoder/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0³
encoder/conv2d_12/BiasAddBiasAdd!encoder/conv2d_12/Conv2D:output:00encoder/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
encoder/conv2d_12/ReluRelu"encoder/conv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'encoder/conv2d_13/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Û
encoder/conv2d_13/Conv2DConv2D$encoder/conv2d_12/Relu:activations:0/encoder/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

(encoder/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0³
encoder/conv2d_13/BiasAddBiasAdd!encoder/conv2d_13/Conv2D:output:00encoder/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ |
encoder/conv2d_13/ReluRelu"encoder/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'encoder/conv2d_14/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Û
encoder/conv2d_14/Conv2DConv2D$encoder/conv2d_13/Relu:activations:0/encoder/conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

(encoder/conv2d_14/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0³
encoder/conv2d_14/BiasAddBiasAdd!encoder/conv2d_14/Conv2D:output:00encoder/conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
encoder/conv2d_14/ReluRelu"encoder/conv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ 
'encoder/conv2d_15/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0Û
encoder/conv2d_15/Conv2DConv2D$encoder/conv2d_14/Relu:activations:0/encoder/conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

(encoder/conv2d_15/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0³
encoder/conv2d_15/BiasAddBiasAdd!encoder/conv2d_15/Conv2D:output:00encoder/conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH|
encoder/conv2d_15/ReluRelu"encoder/conv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
,encoder/batch_normalization_3/ReadVariableOpReadVariableOp5encoder_batch_normalization_3_readvariableop_resource*
_output_shapes
:H*
dtype0¢
.encoder/batch_normalization_3/ReadVariableOp_1ReadVariableOp7encoder_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:H*
dtype0À
=encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpFencoder_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0Ä
?encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHencoder_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0ê
.encoder/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_15/Relu:activations:04encoder/batch_normalization_3/ReadVariableOp:value:06encoder/batch_normalization_3/ReadVariableOp_1:value:0Eencoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Gencoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
is_training( h
encoder/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ­
encoder/flatten_3/ReshapeReshape2encoder/batch_normalization_3/FusedBatchNormV3:y:0 encoder/flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
&encoder/dense_26/MatMul/ReadVariableOpReadVariableOp/encoder_dense_26_matmul_readvariableop_resource* 
_output_shapes
:
	*
dtype0¨
encoder/dense_26/MatMulMatMul"encoder/flatten_3/Reshape:output:0.encoder/dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'encoder/dense_26/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
encoder/dense_26/BiasAddBiasAdd!encoder/dense_26/MatMul:product:0/encoder/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
encoder/dense_26/ReluRelu!encoder/dense_26/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&encoder/dense_27/MatMul/ReadVariableOpReadVariableOp/encoder_dense_27_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0¨
encoder/dense_27/MatMulMatMul#encoder/dense_26/Relu:activations:0.encoder/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
'encoder/dense_27/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_27_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0©
encoder/dense_27/BiasAddBiasAdd!encoder/dense_27/MatMul:product:0/encoder/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHz
encoder/dense_27/SoftplusSoftplus!encoder/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0
encoder/z/MatMulMatMul#encoder/dense_26/Relu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHv
IdentityIdentity'encoder/dense_27/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHk

Identity_1Identityencoder/z/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHç
NoOpNoOp>^encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp@^encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1-^encoder/batch_normalization_3/ReadVariableOp/^encoder/batch_normalization_3/ReadVariableOp_1)^encoder/conv2d_12/BiasAdd/ReadVariableOp(^encoder/conv2d_12/Conv2D/ReadVariableOp)^encoder/conv2d_13/BiasAdd/ReadVariableOp(^encoder/conv2d_13/Conv2D/ReadVariableOp)^encoder/conv2d_14/BiasAdd/ReadVariableOp(^encoder/conv2d_14/Conv2D/ReadVariableOp)^encoder/conv2d_15/BiasAdd/ReadVariableOp(^encoder/conv2d_15/Conv2D/ReadVariableOp(^encoder/dense_26/BiasAdd/ReadVariableOp'^encoder/dense_26/MatMul/ReadVariableOp(^encoder/dense_27/BiasAdd/ReadVariableOp'^encoder/dense_27/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2~
=encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp=encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2
?encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1?encoder/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12\
,encoder/batch_normalization_3/ReadVariableOp,encoder/batch_normalization_3/ReadVariableOp2`
.encoder/batch_normalization_3/ReadVariableOp_1.encoder/batch_normalization_3/ReadVariableOp_12T
(encoder/conv2d_12/BiasAdd/ReadVariableOp(encoder/conv2d_12/BiasAdd/ReadVariableOp2R
'encoder/conv2d_12/Conv2D/ReadVariableOp'encoder/conv2d_12/Conv2D/ReadVariableOp2T
(encoder/conv2d_13/BiasAdd/ReadVariableOp(encoder/conv2d_13/BiasAdd/ReadVariableOp2R
'encoder/conv2d_13/Conv2D/ReadVariableOp'encoder/conv2d_13/Conv2D/ReadVariableOp2T
(encoder/conv2d_14/BiasAdd/ReadVariableOp(encoder/conv2d_14/BiasAdd/ReadVariableOp2R
'encoder/conv2d_14/Conv2D/ReadVariableOp'encoder/conv2d_14/Conv2D/ReadVariableOp2T
(encoder/conv2d_15/BiasAdd/ReadVariableOp(encoder/conv2d_15/BiasAdd/ReadVariableOp2R
'encoder/conv2d_15/Conv2D/ReadVariableOp'encoder/conv2d_15/Conv2D/ReadVariableOp2R
'encoder/dense_26/BiasAdd/ReadVariableOp'encoder/dense_26/BiasAdd/ReadVariableOp2P
&encoder/dense_26/MatMul/ReadVariableOp&encoder/dense_26/MatMul/ReadVariableOp2R
'encoder/dense_27/BiasAdd/ReadVariableOp'encoder/dense_27/BiasAdd/ReadVariableOp2P
&encoder/dense_27/MatMul/ReadVariableOp&encoder/dense_27/MatMul/ReadVariableOp2D
 encoder/z/BiasAdd/ReadVariableOp encoder/z/BiasAdd/ReadVariableOp2B
encoder/z/MatMul/ReadVariableOpencoder/z/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9
Ø
ù
(__inference_encoder_layer_call_fn_496667

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
	

unknown_12:	

unknown_13:	H

unknown_14:H

unknown_15:	H

unknown_16:H
identity

identity_1¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_495883o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
E__inference_conv2d_13_layer_call_and_return_conditional_losses_497038

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
E__inference_conv2d_14_layer_call_and_return_conditional_losses_497073

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Þ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

å
E__inference_conv2d_12_layer_call_and_return_conditional_losses_497003

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_497181

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
ï

*__inference_conv2d_15_layer_call_fn_497082

inputs!
unknown:@H
	unknown_0:H
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
J
¿
"__inference__traced_restore_497462
file_prefix;
!assignvariableop_conv2d_12_kernel:/
!assignvariableop_1_conv2d_12_bias:=
#assignvariableop_2_conv2d_13_kernel: /
!assignvariableop_3_conv2d_13_bias: =
#assignvariableop_4_conv2d_14_kernel: @/
!assignvariableop_5_conv2d_14_bias:@=
#assignvariableop_6_conv2d_15_kernel:@H/
!assignvariableop_7_conv2d_15_bias:H<
.assignvariableop_8_batch_normalization_3_gamma:H;
-assignvariableop_9_batch_normalization_3_beta:HC
5assignvariableop_10_batch_normalization_3_moving_mean:HG
9assignvariableop_11_batch_normalization_3_moving_variance:H7
#assignvariableop_12_dense_26_kernel:
	0
!assignvariableop_13_dense_26_bias:	/
assignvariableop_14_z_kernel:	H(
assignvariableop_15_z_bias:H6
#assignvariableop_16_dense_27_kernel:	H/
!assignvariableop_17_dense_27_bias:H
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ú
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_14_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_14_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_15_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_15_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_26_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_26_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_z_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_z_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_27_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_27_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Û
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: È
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
«
Í
C__inference_encoder_layer_call_and_return_conditional_losses_496968

inputsB
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource: 7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: @7
)conv2d_14_biasadd_readvariableop_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@H7
)conv2d_15_biasadd_readvariableop_resource:H;
-batch_normalization_3_readvariableop_resource:H=
/batch_normalization_3_readvariableop_1_resource:HL
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:H;
'dense_26_matmul_readvariableop_resource:
	7
(dense_26_biasadd_readvariableop_resource:	:
'dense_27_matmul_readvariableop_resource:	H6
(dense_27_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	H/
!z_biasadd_readvariableop_resource:H
identity

identity_1¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_13/BiasAdd/ReadVariableOp¢conv2d_13/Conv2D/ReadVariableOp¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_14/BiasAdd/ReadVariableOp¢conv2d_14/Conv2D/ReadVariableOp¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_15/BiasAdd/ReadVariableOp¢conv2d_15/Conv2D/ReadVariableOp¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢dense_27/BiasAdd/ReadVariableOp¢dense_27/MatMul/ReadVariableOp¢z/BiasAdd/ReadVariableOp¢z/MatMul/ReadVariableOp
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ã
conv2d_14/Conv2DConv2Dconv2d_13/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0Ã
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHl
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:H*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:H*
dtype0°
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0´
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0È
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
exponential_avg_factor%
×#<
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¨
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_3/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
	*
dtype0
dense_26/MatMulMatMulflatten_3/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHj
dense_27/SoftplusSoftplusdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0
z/MatMulMatMuldense_26/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHg
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHp

Identity_1Identitydense_27/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHÃ	
NoOpNoOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_3_layer_call_fn_497121

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495577
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497170

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿHÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs

å
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHg
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHÞ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
Î
C__inference_encoder_layer_call_and_return_conditional_losses_496349
input_9*
conv2d_12_496242:
conv2d_12_496244:*
conv2d_13_496247: 
conv2d_13_496249: *
conv2d_14_496252: @
conv2d_14_496254:@*
conv2d_15_496257:@H
conv2d_15_496259:H*
batch_normalization_3_496262:H*
batch_normalization_3_496264:H*
batch_normalization_3_496266:H*
batch_normalization_3_496268:H#
dense_26_496272:
	
dense_26_496274:	"
dense_27_496277:	H
dense_27_496279:H
z_496282:	H
z_496284:H
identity

identity_1¢-batch_normalization_3/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_13/StatefulPartitionedCall¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_14/StatefulPartitionedCall¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_15/StatefulPartitionedCall¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢z/StatefulPartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_12_496242conv2d_12_496244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652£
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_496247conv2d_13_496249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684£
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_496252conv2d_14_496254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716£
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_496257conv2d_15_496259*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_3_496262batch_normalization_3_496264batch_normalization_3_496266batch_normalization_3_496268*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495577î
flatten_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_26_496272dense_26_496274*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_495782
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_496277dense_27_496279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_495799ú
z/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0z_496282z_496284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_495815g
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_12_496242*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_496242*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_13_496247*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_496247*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_14_496252*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_496252*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_15_496257*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_496257*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHz

Identity_1Identity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp"^conv2d_15/StatefulPartitionedCall0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9
Ì

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497152

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
«

ö
D__inference_dense_27_layer_call_and_return_conditional_losses_497240

inputs1
matmul_readvariableop_resource:	H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHe
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä	
ï
=__inference_z_layer_call_and_return_conditional_losses_497220

inputs1
matmul_readvariableop_resource:	H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_3_layer_call_fn_497134

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495608
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Ä	
ï
=__inference_z_layer_call_and_return_conditional_losses_495815

inputs1
matmul_readvariableop_resource:	H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í£
ý
C__inference_encoder_layer_call_and_return_conditional_losses_496839

inputsB
(conv2d_12_conv2d_readvariableop_resource:7
)conv2d_12_biasadd_readvariableop_resource:B
(conv2d_13_conv2d_readvariableop_resource: 7
)conv2d_13_biasadd_readvariableop_resource: B
(conv2d_14_conv2d_readvariableop_resource: @7
)conv2d_14_biasadd_readvariableop_resource:@B
(conv2d_15_conv2d_readvariableop_resource:@H7
)conv2d_15_biasadd_readvariableop_resource:H;
-batch_normalization_3_readvariableop_resource:H=
/batch_normalization_3_readvariableop_1_resource:HL
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:H;
'dense_26_matmul_readvariableop_resource:
	7
(dense_26_biasadd_readvariableop_resource:	:
'dense_27_matmul_readvariableop_resource:	H6
(dense_27_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	H/
!z_biasadd_readvariableop_resource:H
identity

identity_1¢5batch_normalization_3/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_3/ReadVariableOp¢&batch_normalization_3/ReadVariableOp_1¢ conv2d_12/BiasAdd/ReadVariableOp¢conv2d_12/Conv2D/ReadVariableOp¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_13/BiasAdd/ReadVariableOp¢conv2d_13/Conv2D/ReadVariableOp¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_14/BiasAdd/ReadVariableOp¢conv2d_14/Conv2D/ReadVariableOp¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢ conv2d_15/BiasAdd/ReadVariableOp¢conv2d_15/Conv2D/ReadVariableOp¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢dense_26/BiasAdd/ReadVariableOp¢dense_26/MatMul/ReadVariableOp¢dense_27/BiasAdd/ReadVariableOp¢dense_27/MatMul/ReadVariableOp¢z/BiasAdd/ReadVariableOp¢z/MatMul/ReadVariableOp
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0­
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_13/Conv2DConv2Dconv2d_12/Relu:activations:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ l
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_14/Conv2D/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ã
conv2d_14/Conv2DConv2Dconv2d_13/Relu:activations:0'conv2d_14/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_14/BiasAdd/ReadVariableOpReadVariableOp)conv2d_14_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_14/BiasAddBiasAddconv2d_14/Conv2D:output:0(conv2d_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@l
conv2d_14/ReluReluconv2d_14/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_15/Conv2D/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0Ã
conv2d_15/Conv2DConv2Dconv2d_14/Relu:activations:0'conv2d_15/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*
paddingSAME*
strides

 conv2d_15/BiasAdd/ReadVariableOpReadVariableOp)conv2d_15_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
conv2d_15/BiasAddBiasAddconv2d_15/Conv2D:output:0(conv2d_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿHl
conv2d_15/ReluReluconv2d_15/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:H*
dtype0
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:H*
dtype0°
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0´
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0º
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_15/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
is_training( `
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_3/ReshapeReshape*batch_normalization_3/FusedBatchNormV3:y:0flatten_3/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
	*
dtype0
dense_26/MatMulMatMulflatten_3/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHj
dense_27/SoftplusSoftplusdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	H*
dtype0
z/MatMulMatMuldense_26/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHg
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_14_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: £
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_15_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHp

Identity_1Identitydense_27/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHó
NoOpNoOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp!^conv2d_14/BiasAdd/ReadVariableOp ^conv2d_14/Conv2D/ReadVariableOp0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp!^conv2d_15/BiasAdd/ReadVariableOp ^conv2d_15/Conv2D/ReadVariableOp0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_14/BiasAdd/ReadVariableOp conv2d_14/BiasAdd/ReadVariableOp2B
conv2d_14/Conv2D/ReadVariableOpconv2d_14/Conv2D/ReadVariableOp2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_15/BiasAdd/ReadVariableOp conv2d_15/BiasAdd/ReadVariableOp2B
conv2d_15/Conv2D/ReadVariableOpconv2d_15/Conv2D/ReadVariableOp2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
ö
$__inference_signature_wrapper_496564
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
	

unknown_12:	

unknown_13:	H

unknown_14:H

unknown_15:	H

unknown_16:H
identity

identity_1¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_495555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9
Ö
ù
(__inference_encoder_layer_call_fn_496710

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
	

unknown_12:	

unknown_13:	H

unknown_14:H

unknown_15:	H

unknown_16:H
identity

identity_1¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_496155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
Í
C__inference_encoder_layer_call_and_return_conditional_losses_495883

inputs*
conv2d_12_495653:
conv2d_12_495655:*
conv2d_13_495685: 
conv2d_13_495687: *
conv2d_14_495717: @
conv2d_14_495719:@*
conv2d_15_495749:@H
conv2d_15_495751:H*
batch_normalization_3_495754:H*
batch_normalization_3_495756:H*
batch_normalization_3_495758:H*
batch_normalization_3_495760:H#
dense_26_495783:
	
dense_26_495785:	"
dense_27_495800:	H
dense_27_495802:H
z_495816:	H
z_495818:H
identity

identity_1¢-batch_normalization_3/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_13/StatefulPartitionedCall¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_14/StatefulPartitionedCall¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_15/StatefulPartitionedCall¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢z/StatefulPartitionedCallÿ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_495653conv2d_12_495655*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652£
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_495685conv2d_13_495687*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684£
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_495717conv2d_14_495719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716£
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_495749conv2d_15_495751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_3_495754batch_normalization_3_495756batch_normalization_3_495758batch_normalization_3_495760*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495577î
flatten_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_26_495783dense_26_495785*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_495782
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_495800dense_27_495802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_495799ú
z/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0z_495816z_495818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_495815g
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_12_495653*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_495653*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_13_495685*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_495685*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_14_495717*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_495717*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_15_495749*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_495749*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHz

Identity_1Identity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp"^conv2d_15/StatefulPartitionedCall0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495577

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH:H:H:H:H:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
ï

*__inference_conv2d_13_layer_call_fn_497012

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ø
D__inference_dense_26_layer_call_and_return_conditional_losses_495782

inputs2
matmul_readvariableop_resource:
	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
É
Í
C__inference_encoder_layer_call_and_return_conditional_losses_496155

inputs*
conv2d_12_496048:
conv2d_12_496050:*
conv2d_13_496053: 
conv2d_13_496055: *
conv2d_14_496058: @
conv2d_14_496060:@*
conv2d_15_496063:@H
conv2d_15_496065:H*
batch_normalization_3_496068:H*
batch_normalization_3_496070:H*
batch_normalization_3_496072:H*
batch_normalization_3_496074:H#
dense_26_496078:
	
dense_26_496080:	"
dense_27_496083:	H
dense_27_496085:H
z_496088:	H
z_496090:H
identity

identity_1¢-batch_normalization_3/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_13/StatefulPartitionedCall¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_14/StatefulPartitionedCall¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_15/StatefulPartitionedCall¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢z/StatefulPartitionedCallÿ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_496048conv2d_12_496050*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652£
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_496053conv2d_13_496055*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684£
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_496058conv2d_14_496060*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716£
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_496063conv2d_15_496065*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_3_496068batch_normalization_3_496070batch_normalization_3_496072batch_normalization_3_496074*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495608î
flatten_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_26_496078dense_26_496080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_495782
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_496083dense_27_496085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_495799ú
z/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0z_496088z_496090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_495815g
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_12_496048*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_496048*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_13_496053*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_496053*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_14_496058*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_496058*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_15_496063*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_496063*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHz

Identity_1Identity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp"^conv2d_15/StatefulPartitionedCall0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

*__inference_conv2d_12_layer_call_fn_496977

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
ê
__inference_loss_fn_0_497260R
8conv2d_12_kernel_regularizer_abs_readvariableop_resource:
identity¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_12_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ³
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_12_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_12/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp
Ì

)__inference_dense_26_layer_call_fn_497190

inputs
unknown:
	
	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_495782p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Û
ú
(__inference_encoder_layer_call_fn_495924
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
	

unknown_12:	

unknown_13:	H

unknown_14:H

unknown_15:	H

unknown_16:H
identity

identity_1¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_495883o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9
Ì
Î
C__inference_encoder_layer_call_and_return_conditional_losses_496459
input_9*
conv2d_12_496352:
conv2d_12_496354:*
conv2d_13_496357: 
conv2d_13_496359: *
conv2d_14_496362: @
conv2d_14_496364:@*
conv2d_15_496367:@H
conv2d_15_496369:H*
batch_normalization_3_496372:H*
batch_normalization_3_496374:H*
batch_normalization_3_496376:H*
batch_normalization_3_496378:H#
dense_26_496382:
	
dense_26_496384:	"
dense_27_496387:	H
dense_27_496389:H
z_496392:	H
z_496394:H
identity

identity_1¢-batch_normalization_3/StatefulPartitionedCall¢!conv2d_12/StatefulPartitionedCall¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_13/StatefulPartitionedCall¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_14/StatefulPartitionedCall¢/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_14/kernel/Regularizer/Square/ReadVariableOp¢!conv2d_15/StatefulPartitionedCall¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_26/StatefulPartitionedCall¢ dense_27/StatefulPartitionedCall¢z/StatefulPartitionedCall
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_12_496352conv2d_12_496354*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652£
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0conv2d_13_496357conv2d_13_496359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_13_layer_call_and_return_conditional_losses_495684£
!conv2d_14/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0conv2d_14_496362conv2d_14_496364*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716£
!conv2d_15/StatefulPartitionedCallStatefulPartitionedCall*conv2d_14/StatefulPartitionedCall:output:0conv2d_15_496367conv2d_15_496369*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_15_layer_call_and_return_conditional_losses_495748
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*conv2d_15/StatefulPartitionedCall:output:0batch_normalization_3_496372batch_normalization_3_496374batch_normalization_3_496376batch_normalization_3_496378*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_495608î
flatten_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769
 dense_26/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_26_496382dense_26_496384*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_26_layer_call_and_return_conditional_losses_495782
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_496387dense_27_496389*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_27_layer_call_and_return_conditional_losses_495799ú
z/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0z_496392z_496394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_495815g
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_12_496352*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_12_496352*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_13_496357*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_13_496357*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_14_496362*&
_output_shapes
: @*
dtype0
 conv2d_14/kernel/Regularizer/AbsAbs7conv2d_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_14/kernel/Regularizer/SumSum$conv2d_14/kernel/Regularizer/Abs:y:0-conv2d_14/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_14/kernel/Regularizer/mulMul+conv2d_14/kernel/Regularizer/mul/x:output:0)conv2d_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_14/kernel/Regularizer/addAddV2+conv2d_14/kernel/Regularizer/Const:output:0$conv2d_14/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_14/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_14_496362*&
_output_shapes
: @*
dtype0
#conv2d_14/kernel/Regularizer/SquareSquare:conv2d_14/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_14/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_14/kernel/Regularizer/Sum_1Sum'conv2d_14/kernel/Regularizer/Square:y:0-conv2d_14/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_14/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_14/kernel/Regularizer/mul_1Mul-conv2d_14/kernel/Regularizer/mul_1/x:output:0+conv2d_14/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_14/kernel/Regularizer/add_1AddV2$conv2d_14/kernel/Regularizer/add:z:0&conv2d_14/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_15_496367*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_15_496367*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHz

Identity_1Identity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp"^conv2d_13/StatefulPartitionedCall0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp"^conv2d_14/StatefulPartitionedCall0^conv2d_14/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_14/kernel/Regularizer/Square/ReadVariableOp"^conv2d_15/StatefulPartitionedCall0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_14/StatefulPartitionedCall!conv2d_14/StatefulPartitionedCall2b
/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp/conv2d_14/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2conv2d_14/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_15/StatefulPartitionedCall!conv2d_15/StatefulPartitionedCall2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9
§

ø
D__inference_dense_26_layer_call_and_return_conditional_losses_497201

inputs2
matmul_readvariableop_resource:
	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ï

*__inference_conv2d_14_layer_call_fn_497047

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_14_layer_call_and_return_conditional_losses_495716w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ô
ê
__inference_loss_fn_3_497320R
8conv2d_15_kernel_regularizer_abs_readvariableop_resource:@H
identity¢/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_15/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_15/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_15_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0
 conv2d_15/kernel/Regularizer/AbsAbs7conv2d_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_15/kernel/Regularizer/SumSum$conv2d_15/kernel/Regularizer/Abs:y:0-conv2d_15/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_15/kernel/Regularizer/mulMul+conv2d_15/kernel/Regularizer/mul/x:output:0)conv2d_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_15/kernel/Regularizer/addAddV2+conv2d_15/kernel/Regularizer/Const:output:0$conv2d_15/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ³
2conv2d_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_15_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0
#conv2d_15/kernel/Regularizer/SquareSquare:conv2d_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_15/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_15/kernel/Regularizer/Sum_1Sum'conv2d_15/kernel/Regularizer/Square:y:0-conv2d_15/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_15/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_15/kernel/Regularizer/mul_1Mul-conv2d_15/kernel/Regularizer/mul_1/x:output:0+conv2d_15/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_15/kernel/Regularizer/add_1AddV2$conv2d_15/kernel/Regularizer/add:z:0&conv2d_15/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_15/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^conv2d_15/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_15/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp/conv2d_15/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_15/kernel/Regularizer/Square/ReadVariableOp2conv2d_15/kernel/Regularizer/Square/ReadVariableOp
Ç
a
E__inference_flatten_3_layer_call_and_return_conditional_losses_495769

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿH:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿH
 
_user_specified_nameinputs
Ù-
÷
__inference__traced_save_497398
file_prefix/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop/
+savev2_conv2d_14_kernel_read_readvariableop-
)savev2_conv2d_14_bias_read_readvariableop/
+savev2_conv2d_15_kernel_read_readvariableop-
)savev2_conv2d_15_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ÷
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop+savev2_conv2d_14_kernel_read_readvariableop)savev2_conv2d_14_bias_read_readvariableop+savev2_conv2d_15_kernel_read_readvariableop)savev2_conv2d_15_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Æ
_input_shapes´
±: ::: : : @:@:@H:H:H:H:H:H:
	::	H:H:	H:H: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@H: 

_output_shapes
:H: 	

_output_shapes
:H: 


_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H:&"
 
_output_shapes
:
	:!

_output_shapes	
::%!

_output_shapes
:	H: 

_output_shapes
:H:%!

_output_shapes
:	H: 

_output_shapes
:H:

_output_shapes
: 
«

ö
D__inference_dense_27_layer_call_and_return_conditional_losses_495799

inputs1
matmul_readvariableop_resource:	H-
biasadd_readvariableop_resource:H
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHe
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

å
E__inference_conv2d_12_layer_call_and_return_conditional_losses_495652

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp¢/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_12/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"conv2d_12/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
 conv2d_12/kernel/Regularizer/AbsAbs7conv2d_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_12/kernel/Regularizer/SumSum$conv2d_12/kernel/Regularizer/Abs:y:0-conv2d_12/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_12/kernel/Regularizer/mulMul+conv2d_12/kernel/Regularizer/mul/x:output:0)conv2d_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_12/kernel/Regularizer/addAddV2+conv2d_12/kernel/Regularizer/Const:output:0$conv2d_12/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 
2conv2d_12/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_12/kernel/Regularizer/SquareSquare:conv2d_12/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:}
$conv2d_12/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_12/kernel/Regularizer/Sum_1Sum'conv2d_12/kernel/Regularizer/Square:y:0-conv2d_12/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_12/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_12/kernel/Regularizer/mul_1Mul-conv2d_12/kernel/Regularizer/mul_1/x:output:0+conv2d_12/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_12/kernel/Regularizer/add_1AddV2$conv2d_12/kernel/Regularizer/add:z:0&conv2d_12/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_12/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_12/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp/conv2d_12/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_12/kernel/Regularizer/Square/ReadVariableOp2conv2d_12/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
ê
__inference_loss_fn_1_497280R
8conv2d_13_kernel_regularizer_abs_readvariableop_resource: 
identity¢/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp¢2conv2d_13/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_13/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    °
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_13_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0
 conv2d_13/kernel/Regularizer/AbsAbs7conv2d_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
 conv2d_13/kernel/Regularizer/SumSum$conv2d_13/kernel/Regularizer/Abs:y:0-conv2d_13/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#< 
 conv2d_13/kernel/Regularizer/mulMul+conv2d_13/kernel/Regularizer/mul/x:output:0)conv2d_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 
 conv2d_13/kernel/Regularizer/addAddV2+conv2d_13/kernel/Regularizer/Const:output:0$conv2d_13/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ³
2conv2d_13/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_13_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0
#conv2d_13/kernel/Regularizer/SquareSquare:conv2d_13/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: }
$conv2d_13/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ¢
"conv2d_13/kernel/Regularizer/Sum_1Sum'conv2d_13/kernel/Regularizer/Square:y:0-conv2d_13/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_13/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¦
"conv2d_13/kernel/Regularizer/mul_1Mul-conv2d_13/kernel/Regularizer/mul_1/x:output:0+conv2d_13/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 
"conv2d_13/kernel/Regularizer/add_1AddV2$conv2d_13/kernel/Regularizer/add:z:0&conv2d_13/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_13/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ­
NoOpNoOp0^conv2d_13/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_13/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp/conv2d_13/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_13/kernel/Regularizer/Square/ReadVariableOp2conv2d_13/kernel/Regularizer/Square/ReadVariableOp
Ù
ú
(__inference_encoder_layer_call_fn_496239
input_9!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
	

unknown_12:	

unknown_13:	H

unknown_14:H

unknown_15:	H

unknown_16:H
identity

identity_1¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿH:ÿÿÿÿÿÿÿÿÿH*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_496155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿHq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_9"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ê
serving_defaultÖ
C
input_98
serving_default_input_9:0ÿÿÿÿÿÿÿÿÿ<
dense_270
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿH5
z0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿHtensorflow/serving/predict:÷ò
¦
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ý
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ý
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
Ý
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
Ý
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
ê
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
¥
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
»
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
»
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
»
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias"
_tf_keras_layer
¦
0
1
"2
#3
+4
,5
46
57
>8
?9
@10
A11
N12
O13
V14
W15
^16
_17"
trackable_list_wrapper

0
1
"2
#3
+4
,5
46
57
>8
?9
N10
O11
V12
W13
^14
_15"
trackable_list_wrapper
<
`0
a1
b2
c3"
trackable_list_wrapper
Ê
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö
itrace_0
jtrace_1
ktrace_2
ltrace_32ë
(__inference_encoder_layer_call_fn_495924
(__inference_encoder_layer_call_fn_496667
(__inference_encoder_layer_call_fn_496710
(__inference_encoder_layer_call_fn_496239À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zitrace_0zjtrace_1zktrace_2zltrace_3
Â
mtrace_0
ntrace_1
otrace_2
ptrace_32×
C__inference_encoder_layer_call_and_return_conditional_losses_496839
C__inference_encoder_layer_call_and_return_conditional_losses_496968
C__inference_encoder_layer_call_and_return_conditional_losses_496349
C__inference_encoder_layer_call_and_return_conditional_losses_496459À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zmtrace_0zntrace_1zotrace_2zptrace_3
ÌBÉ
!__inference__wrapped_model_495555input_9"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
qserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
wtrace_02Ñ
*__inference_conv2d_12_layer_call_fn_496977¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zwtrace_0

xtrace_02ì
E__inference_conv2d_12_layer_call_and_return_conditional_losses_497003¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zxtrace_0
*:(2conv2d_12/kernel
:2conv2d_12/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
'
a0"
trackable_list_wrapper
­
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
î
~trace_02Ñ
*__inference_conv2d_13_layer_call_fn_497012¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z~trace_0

trace_02ì
E__inference_conv2d_13_layer_call_and_return_conditional_losses_497038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
*:( 2conv2d_13/kernel
: 2conv2d_13/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_conv2d_14_layer_call_fn_497047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_conv2d_14_layer_call_and_return_conditional_losses_497073¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
*:( @2conv2d_14/kernel
:@2conv2d_14/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
'
c0"
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_conv2d_15_layer_call_fn_497082¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_conv2d_15_layer_call_and_return_conditional_losses_497108¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
*:(@H2conv2d_15/kernel
:H2conv2d_15/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
â
trace_0
trace_12§
6__inference_batch_normalization_3_layer_call_fn_497121
6__inference_batch_normalization_3_layer_call_fn_497134´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ý
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497152
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497170´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
):'H2batch_normalization_3/gamma
(:&H2batch_normalization_3/beta
1:/H (2!batch_normalization_3/moving_mean
5:3H (2%batch_normalization_3/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_flatten_3_layer_call_fn_497175¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_flatten_3_layer_call_and_return_conditional_losses_497181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ï
£trace_02Ð
)__inference_dense_26_layer_call_fn_497190¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z£trace_0

¤trace_02ë
D__inference_dense_26_layer_call_and_return_conditional_losses_497201¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¤trace_0
#:!
	2dense_26/kernel
:2dense_26/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
è
ªtrace_02É
"__inference_z_layer_call_fn_497210¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zªtrace_0

«trace_02ä
=__inference_z_layer_call_and_return_conditional_losses_497220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z«trace_0
:	H2z/kernel
:H2z/bias
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
ï
±trace_02Ð
)__inference_dense_27_layer_call_fn_497229¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z±trace_0

²trace_02ë
D__inference_dense_27_layer_call_and_return_conditional_losses_497240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z²trace_0
": 	H2dense_27/kernel
:H2dense_27/bias
Ï
³trace_02°
__inference_loss_fn_0_497260
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z³trace_0
Ï
´trace_02°
__inference_loss_fn_1_497280
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z´trace_0
Ï
µtrace_02°
__inference_loss_fn_2_497300
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ zµtrace_0
Ï
¶trace_02°
__inference_loss_fn_3_497320
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ z¶trace_0
.
@0
A1"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ûBø
(__inference_encoder_layer_call_fn_495924input_9"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_encoder_layer_call_fn_496667inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
(__inference_encoder_layer_call_fn_496710inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ûBø
(__inference_encoder_layer_call_fn_496239input_9"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_496839inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_496968inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_496349input_9"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
C__inference_encoder_layer_call_and_return_conditional_losses_496459input_9"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_496564input_9"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv2d_12_layer_call_fn_496977inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv2d_12_layer_call_and_return_conditional_losses_497003inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
a0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv2d_13_layer_call_fn_497012inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv2d_13_layer_call_and_return_conditional_losses_497038inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv2d_14_layer_call_fn_497047inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv2d_14_layer_call_and_return_conditional_losses_497073inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
c0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_conv2d_15_layer_call_fn_497082inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_conv2d_15_layer_call_and_return_conditional_losses_497108inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_3_layer_call_fn_497121inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_3_layer_call_fn_497134inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497152inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497170inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
ÞBÛ
*__inference_flatten_3_layer_call_fn_497175inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_flatten_3_layer_call_and_return_conditional_losses_497181inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_dense_26_layer_call_fn_497190inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_26_layer_call_and_return_conditional_losses_497201inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÖBÓ
"__inference_z_layer_call_fn_497210inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñBî
=__inference_z_layer_call_and_return_conditional_losses_497220inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÝBÚ
)__inference_dense_27_layer_call_fn_497229inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_dense_27_layer_call_and_return_conditional_losses_497240inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³B°
__inference_loss_fn_0_497260"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³B°
__inference_loss_fn_1_497280"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³B°
__inference_loss_fn_2_497300"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³B°
__inference_loss_fn_3_497320"
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ Ë
!__inference__wrapped_model_495555¥"#+,45>?@ANO^_VW8¢5
.¢+
)&
input_9ÿÿÿÿÿÿÿÿÿ
ª "UªR
.
dense_27"
dense_27ÿÿÿÿÿÿÿÿÿH
 
z
zÿÿÿÿÿÿÿÿÿHì
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497152>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 ì
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_497170>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
 Ä
6__inference_batch_normalization_3_layer_call_fn_497121>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿHÄ
6__inference_batch_normalization_3_layer_call_fn_497134>?@AM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿHµ
E__inference_conv2d_12_layer_call_and_return_conditional_losses_497003l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv2d_12_layer_call_fn_496977_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_13_layer_call_and_return_conditional_losses_497038l"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_conv2d_13_layer_call_fn_497012_"#7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2d_14_layer_call_and_return_conditional_losses_497073l+,7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_14_layer_call_fn_497047_+,7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@µ
E__inference_conv2d_15_layer_call_and_return_conditional_losses_497108l457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿH
 
*__inference_conv2d_15_layer_call_fn_497082_457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿH¦
D__inference_dense_26_layer_call_and_return_conditional_losses_497201^NO0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ	
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_26_layer_call_fn_497190QNO0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ	
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_27_layer_call_and_return_conditional_losses_497240]^_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿH
 }
)__inference_dense_27_layer_call_fn_497229P^_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿHë
C__inference_encoder_layer_call_and_return_conditional_losses_496349£"#+,45>?@ANO^_VW@¢=
6¢3
)&
input_9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿH

0/1ÿÿÿÿÿÿÿÿÿH
 ë
C__inference_encoder_layer_call_and_return_conditional_losses_496459£"#+,45>?@ANO^_VW@¢=
6¢3
)&
input_9ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿH

0/1ÿÿÿÿÿÿÿÿÿH
 ê
C__inference_encoder_layer_call_and_return_conditional_losses_496839¢"#+,45>?@ANO^_VW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿH

0/1ÿÿÿÿÿÿÿÿÿH
 ê
C__inference_encoder_layer_call_and_return_conditional_losses_496968¢"#+,45>?@ANO^_VW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿH

0/1ÿÿÿÿÿÿÿÿÿH
 Â
(__inference_encoder_layer_call_fn_495924"#+,45>?@ANO^_VW@¢=
6¢3
)&
input_9ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿH

1ÿÿÿÿÿÿÿÿÿHÂ
(__inference_encoder_layer_call_fn_496239"#+,45>?@ANO^_VW@¢=
6¢3
)&
input_9ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿH

1ÿÿÿÿÿÿÿÿÿHÁ
(__inference_encoder_layer_call_fn_496667"#+,45>?@ANO^_VW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿH

1ÿÿÿÿÿÿÿÿÿHÁ
(__inference_encoder_layer_call_fn_496710"#+,45>?@ANO^_VW?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿH

1ÿÿÿÿÿÿÿÿÿHª
E__inference_flatten_3_layer_call_and_return_conditional_losses_497181a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ	
 
*__inference_flatten_3_layer_call_fn_497175T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿH
ª "ÿÿÿÿÿÿÿÿÿ	;
__inference_loss_fn_0_497260¢

¢ 
ª " ;
__inference_loss_fn_1_497280"¢

¢ 
ª " ;
__inference_loss_fn_2_497300+¢

¢ 
ª " ;
__inference_loss_fn_3_4973204¢

¢ 
ª " Ù
$__inference_signature_wrapper_496564°"#+,45>?@ANO^_VWC¢@
¢ 
9ª6
4
input_9)&
input_9ÿÿÿÿÿÿÿÿÿ"UªR
.
dense_27"
dense_27ÿÿÿÿÿÿÿÿÿH
 
z
zÿÿÿÿÿÿÿÿÿH
=__inference_z_layer_call_and_return_conditional_losses_497220]VW0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿH
 v
"__inference_z_layer_call_fn_497210PVW0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿH