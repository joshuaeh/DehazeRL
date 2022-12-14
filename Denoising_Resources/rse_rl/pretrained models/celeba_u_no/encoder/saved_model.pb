??
??
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
?
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
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
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
 ?"serve*2.9.22v2.9.1-132-g18960c44ad38??
r
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_21/bias
k
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes
:H*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	?H*
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
shape:	?H*
shared_name
z/kernel
f
z/kernel/Read/ReadVariableOpReadVariableOpz/kernel*
_output_shapes
:	?H*
dtype0
s
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_20/bias
l
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes	
:?*
dtype0
|
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?* 
shared_namedense_20/kernel
u
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel* 
_output_shapes
:
?	?*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:H*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:H*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:H*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:H*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:H*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:@H*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:@*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
: *
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
?C
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?C
value?CB?C B?C
?
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
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
?
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
?
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
?
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
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses* 
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias*
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias*
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias*
?
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
?
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
?
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
_Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

"0
#1*

"0
#1*
	
a0* 
?
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
_Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

+0
,1*

+0
,1*
	
b0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

40
51*

40
51*
	
c0* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

N0
O1*

N0
O1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

?trace_0* 

?trace_0* 

?trace_0* 

?trace_0* 
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
?
serving_default_input_7Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_20/kerneldense_20/biasdense_21/kerneldense_21/biasz/kernelz/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_363993
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_364827
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense_20/kerneldense_20/biasz/kernelz/biasdense_21/kerneldense_21/bias*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_364891??
?
F
*__inference_flatten_2_layer_call_fn_364604

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?

?
D__inference_dense_21_layer_call_and_return_conditional_losses_364669

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????HX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:?????????He
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:?????????Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_10_layer_call_fn_364476

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_2_layer_call_fn_364550

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363006?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_364139

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
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
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
&:?????????H:?????????H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_363584o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_364610

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
)__inference_conv2d_9_layer_call_fn_364441

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_363668
input_7!
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
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
&:?????????H:?????????H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_363584o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
)__inference_dense_20_layer_call_fn_364619

inputs
unknown:
?	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_363211p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_364502

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
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
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
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
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?

?
D__inference_dense_21_layer_call_and_return_conditional_losses_363228

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????HX
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:?????????He
IdentityIdentitySoftplus:activations:0^NoOp*
T0*'
_output_shapes
:?????????Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_364268

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: B
(conv2d_10_conv2d_readvariableop_resource: @7
)conv2d_10_biasadd_readvariableop_resource:@B
(conv2d_11_conv2d_readvariableop_resource:@H7
)conv2d_11_biasadd_readvariableop_resource:H;
-batch_normalization_2_readvariableop_resource:H=
/batch_normalization_2_readvariableop_1_resource:HL
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:H;
'dense_20_matmul_readvariableop_resource:
?	?7
(dense_20_biasadd_readvariableop_resource:	?:
'dense_21_matmul_readvariableop_resource:	?H6
(dense_21_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hl
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:H*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( `
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_2/ReshapeReshape*batch_normalization_2/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????	?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0?
dense_20/MatMulMatMulflatten_2/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hj
dense_21/SoftplusSoftplusdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Hy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
z/MatMulMatMuldense_20/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hf
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Hp

Identity_1Identitydense_21/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_363888
input_7)
conv2d_8_363781:
conv2d_8_363783:)
conv2d_9_363786: 
conv2d_9_363788: *
conv2d_10_363791: @
conv2d_10_363793:@*
conv2d_11_363796:@H
conv2d_11_363798:H*
batch_normalization_2_363801:H*
batch_normalization_2_363803:H*
batch_normalization_2_363805:H*
batch_normalization_2_363807:H#
dense_20_363811:
?	?
dense_20_363813:	?"
dense_21_363816:	?H
dense_21_363818:H
z_363821:	?H
z_363823:H
identity

identity_1??-batch_normalization_2/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp?!conv2d_11/StatefulPartitionedCall?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp? conv2d_8/StatefulPartitionedCall?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp? conv2d_9/StatefulPartitionedCall?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_8_363781conv2d_8_363783*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_363786conv2d_9_363788*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_363791conv2d_10_363793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_363796conv2d_11_363798*
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
GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_2_363801batch_normalization_2_363803batch_normalization_2_363805batch_normalization_2_363807*
Tin	
2*
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
GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363037?
flatten_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_20_363811dense_20_363813*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_363211?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_363816dense_21_363818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_363228?
z/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0z_363821z_363823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_363244f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_363781*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_363781*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_363786*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_363786*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_10_363791*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_363791*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_11_363796*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_363796*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hz

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp"^conv2d_11/StatefulPartitionedCall0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
__inference_loss_fn_1_364709Q
7conv2d_9_kernel_regularizer_abs_readvariableop_resource: 
identity??.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_9/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp
?

?
D__inference_dense_20_layer_call_and_return_conditional_losses_364630

inputs2
matmul_readvariableop_resource:
?	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_363312

inputs)
conv2d_8_363082:
conv2d_8_363084:)
conv2d_9_363114: 
conv2d_9_363116: *
conv2d_10_363146: @
conv2d_10_363148:@*
conv2d_11_363178:@H
conv2d_11_363180:H*
batch_normalization_2_363183:H*
batch_normalization_2_363185:H*
batch_normalization_2_363187:H*
batch_normalization_2_363189:H#
dense_20_363212:
?	?
dense_20_363214:	?"
dense_21_363229:	?H
dense_21_363231:H
z_363245:	?H
z_363247:H
identity

identity_1??-batch_normalization_2/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp?!conv2d_11/StatefulPartitionedCall?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp? conv2d_8/StatefulPartitionedCall?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp? conv2d_9/StatefulPartitionedCall?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_363082conv2d_8_363084*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_363114conv2d_9_363116*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_363146conv2d_10_363148*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_363178conv2d_11_363180*
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
GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_2_363183batch_normalization_2_363185batch_normalization_2_363187batch_normalization_2_363189*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363006?
flatten_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_20_363212dense_20_363214*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_363211?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_363229dense_21_363231*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_363228?
z/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0z_363245z_363247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_363244f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_363082*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_363082*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_363114*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_363114*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_10_363146*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_363146*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_11_363178*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_363178*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hz

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp"^conv2d_11/StatefulPartitionedCall0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_364729R
8conv2d_10_kernel_regularizer_abs_readvariableop_resource: @
identity??/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_10_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_10_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_10/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp
?-
?
__inference__traced_save_364827
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_dense_20_kernel_read_readvariableop,
(savev2_dense_20_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop.
*savev2_dense_21_kernel_read_readvariableop,
(savev2_dense_21_bias_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_dense_20_kernel_read_readvariableop(savev2_dense_20_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop*savev2_dense_21_kernel_read_readvariableop(savev2_dense_21_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2?
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
?: ::: : : @:@:@H:H:H:H:H:H:
?	?:?:	?H:H:	?H:H: 2(
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
?	?:!

_output_shapes	
:?:%!

_output_shapes
:	?H: 

_output_shapes
:H:%!

_output_shapes
:	?H: 

_output_shapes
:H:

_output_shapes
: 
?J
?
"__inference__traced_restore_364891
file_prefix:
 assignvariableop_conv2d_8_kernel:.
 assignvariableop_1_conv2d_8_bias:<
"assignvariableop_2_conv2d_9_kernel: .
 assignvariableop_3_conv2d_9_bias: =
#assignvariableop_4_conv2d_10_kernel: @/
!assignvariableop_5_conv2d_10_bias:@=
#assignvariableop_6_conv2d_11_kernel:@H/
!assignvariableop_7_conv2d_11_bias:H<
.assignvariableop_8_batch_normalization_2_gamma:H;
-assignvariableop_9_batch_normalization_2_beta:HC
5assignvariableop_10_batch_normalization_2_moving_mean:HG
9assignvariableop_11_batch_normalization_2_moving_variance:H7
#assignvariableop_12_dense_20_kernel:
?	?0
!assignvariableop_13_dense_20_bias:	?/
assignvariableop_14_z_kernel:	?H(
assignvariableop_15_z_bias:H6
#assignvariableop_16_dense_21_kernel:	?H/
!assignvariableop_17_dense_21_bias:H
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_10_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_10_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_11_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_11_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_20_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_20_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_z_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_z_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_21_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_21_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: ?
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
?
?
)__inference_conv2d_8_layer_call_fn_364406

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
=__inference_z_layer_call_and_return_conditional_losses_364649

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_20_layer_call_and_return_conditional_losses_363211

inputs2
matmul_readvariableop_resource:
?	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_363778
input_7)
conv2d_8_363671:
conv2d_8_363673:)
conv2d_9_363676: 
conv2d_9_363678: *
conv2d_10_363681: @
conv2d_10_363683:@*
conv2d_11_363686:@H
conv2d_11_363688:H*
batch_normalization_2_363691:H*
batch_normalization_2_363693:H*
batch_normalization_2_363695:H*
batch_normalization_2_363697:H#
dense_20_363701:
?	?
dense_20_363703:	?"
dense_21_363706:	?H
dense_21_363708:H
z_363711:	?H
z_363713:H
identity

identity_1??-batch_normalization_2/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp?!conv2d_11/StatefulPartitionedCall?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp? conv2d_8/StatefulPartitionedCall?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp? conv2d_9/StatefulPartitionedCall?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_8_363671conv2d_8_363673*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_363676conv2d_9_363678*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_363681conv2d_10_363683*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_363686conv2d_11_363688*
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
GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_2_363691batch_normalization_2_363693batch_normalization_2_363695batch_normalization_2_363697*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363006?
flatten_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_20_363701dense_20_363703*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_363211?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_363706dense_21_363708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_363228?
z/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0z_363711z_363713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_363244f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_363671*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_363671*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_363676*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_363676*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_10_363681*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_363681*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_11_363686*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_363686*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hz

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp"^conv2d_11/StatefulPartitionedCall0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_364467

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_conv2d_11_layer_call_fn_364511

inputs!
unknown:@H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
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
GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363037

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
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
:?????????HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????Hg
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????H?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_364537

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
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
:?????????HX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????Hg
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????H?
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
=__inference_z_layer_call_and_return_conditional_losses_363244

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Hw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_2_layer_call_fn_364563

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363037?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363006

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?^
?
!__inference__wrapped_model_362984
input_7I
/encoder_conv2d_8_conv2d_readvariableop_resource:>
0encoder_conv2d_8_biasadd_readvariableop_resource:I
/encoder_conv2d_9_conv2d_readvariableop_resource: >
0encoder_conv2d_9_biasadd_readvariableop_resource: J
0encoder_conv2d_10_conv2d_readvariableop_resource: @?
1encoder_conv2d_10_biasadd_readvariableop_resource:@J
0encoder_conv2d_11_conv2d_readvariableop_resource:@H?
1encoder_conv2d_11_biasadd_readvariableop_resource:HC
5encoder_batch_normalization_2_readvariableop_resource:HE
7encoder_batch_normalization_2_readvariableop_1_resource:HT
Fencoder_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:HV
Hencoder_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:HC
/encoder_dense_20_matmul_readvariableop_resource:
?	??
0encoder_dense_20_biasadd_readvariableop_resource:	?B
/encoder_dense_21_matmul_readvariableop_resource:	?H>
0encoder_dense_21_biasadd_readvariableop_resource:H;
(encoder_z_matmul_readvariableop_resource:	?H7
)encoder_z_biasadd_readvariableop_resource:H
identity

identity_1??=encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp??encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?,encoder/batch_normalization_2/ReadVariableOp?.encoder/batch_normalization_2/ReadVariableOp_1?(encoder/conv2d_10/BiasAdd/ReadVariableOp?'encoder/conv2d_10/Conv2D/ReadVariableOp?(encoder/conv2d_11/BiasAdd/ReadVariableOp?'encoder/conv2d_11/Conv2D/ReadVariableOp?'encoder/conv2d_8/BiasAdd/ReadVariableOp?&encoder/conv2d_8/Conv2D/ReadVariableOp?'encoder/conv2d_9/BiasAdd/ReadVariableOp?&encoder/conv2d_9/Conv2D/ReadVariableOp?'encoder/dense_20/BiasAdd/ReadVariableOp?&encoder/dense_20/MatMul/ReadVariableOp?'encoder/dense_21/BiasAdd/ReadVariableOp?&encoder/dense_21/MatMul/ReadVariableOp? encoder/z/BiasAdd/ReadVariableOp?encoder/z/MatMul/ReadVariableOp?
&encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
encoder/conv2d_8/Conv2DConv2Dinput_7.encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
encoder/conv2d_8/BiasAddBiasAdd encoder/conv2d_8/Conv2D:output:0/encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????z
encoder/conv2d_8/ReluRelu!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
&encoder/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
encoder/conv2d_9/Conv2DConv2D#encoder/conv2d_8/Relu:activations:0.encoder/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
'encoder/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
encoder/conv2d_9/BiasAddBiasAdd encoder/conv2d_9/Conv2D:output:0/encoder/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
encoder/conv2d_9/ReluRelu!encoder/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
'encoder/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
encoder/conv2d_10/Conv2DConv2D#encoder/conv2d_9/Relu:activations:0/encoder/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
(encoder/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
encoder/conv2d_10/BiasAddBiasAdd!encoder/conv2d_10/Conv2D:output:00encoder/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@|
encoder/conv2d_10/ReluRelu"encoder/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
'encoder/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
encoder/conv2d_11/Conv2DConv2D$encoder/conv2d_10/Relu:activations:0/encoder/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
(encoder/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
encoder/conv2d_11/BiasAddBiasAdd!encoder/conv2d_11/Conv2D:output:00encoder/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H|
encoder/conv2d_11/ReluRelu"encoder/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H?
,encoder/batch_normalization_2/ReadVariableOpReadVariableOp5encoder_batch_normalization_2_readvariableop_resource*
_output_shapes
:H*
dtype0?
.encoder/batch_normalization_2/ReadVariableOp_1ReadVariableOp7encoder_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
=encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpFencoder_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
?encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHencoder_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
.encoder/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3$encoder/conv2d_11/Relu:activations:04encoder/batch_normalization_2/ReadVariableOp:value:06encoder/batch_normalization_2/ReadVariableOp_1:value:0Eencoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Gencoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( h
encoder/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
encoder/flatten_2/ReshapeReshape2encoder/batch_normalization_2/FusedBatchNormV3:y:0 encoder/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????	?
&encoder/dense_20/MatMul/ReadVariableOpReadVariableOp/encoder_dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0?
encoder/dense_20/MatMulMatMul"encoder/flatten_2/Reshape:output:0.encoder/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
'encoder/dense_20/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
encoder/dense_20/BiasAddBiasAdd!encoder/dense_20/MatMul:product:0/encoder/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
encoder/dense_20/ReluRelu!encoder/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
&encoder/dense_21/MatMul/ReadVariableOpReadVariableOp/encoder_dense_21_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
encoder/dense_21/MatMulMatMul#encoder/dense_20/Relu:activations:0.encoder/dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H?
'encoder/dense_21/BiasAdd/ReadVariableOpReadVariableOp0encoder_dense_21_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
encoder/dense_21/BiasAddBiasAdd!encoder/dense_21/MatMul:product:0/encoder/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hz
encoder/dense_21/SoftplusSoftplus!encoder/dense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
encoder/z/MatMulMatMul#encoder/dense_20/Relu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hv
IdentityIdentity'encoder/dense_21/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:?????????Hk

Identity_1Identityencoder/z/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp>^encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp@^encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1-^encoder/batch_normalization_2/ReadVariableOp/^encoder/batch_normalization_2/ReadVariableOp_1)^encoder/conv2d_10/BiasAdd/ReadVariableOp(^encoder/conv2d_10/Conv2D/ReadVariableOp)^encoder/conv2d_11/BiasAdd/ReadVariableOp(^encoder/conv2d_11/Conv2D/ReadVariableOp(^encoder/conv2d_8/BiasAdd/ReadVariableOp'^encoder/conv2d_8/Conv2D/ReadVariableOp(^encoder/conv2d_9/BiasAdd/ReadVariableOp'^encoder/conv2d_9/Conv2D/ReadVariableOp(^encoder/dense_20/BiasAdd/ReadVariableOp'^encoder/dense_20/MatMul/ReadVariableOp(^encoder/dense_21/BiasAdd/ReadVariableOp'^encoder/dense_21/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2~
=encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp=encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2?
?encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?encoder/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12\
,encoder/batch_normalization_2/ReadVariableOp,encoder/batch_normalization_2/ReadVariableOp2`
.encoder/batch_normalization_2/ReadVariableOp_1.encoder/batch_normalization_2/ReadVariableOp_12T
(encoder/conv2d_10/BiasAdd/ReadVariableOp(encoder/conv2d_10/BiasAdd/ReadVariableOp2R
'encoder/conv2d_10/Conv2D/ReadVariableOp'encoder/conv2d_10/Conv2D/ReadVariableOp2T
(encoder/conv2d_11/BiasAdd/ReadVariableOp(encoder/conv2d_11/BiasAdd/ReadVariableOp2R
'encoder/conv2d_11/Conv2D/ReadVariableOp'encoder/conv2d_11/Conv2D/ReadVariableOp2R
'encoder/conv2d_8/BiasAdd/ReadVariableOp'encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&encoder/conv2d_8/Conv2D/ReadVariableOp&encoder/conv2d_8/Conv2D/ReadVariableOp2R
'encoder/conv2d_9/BiasAdd/ReadVariableOp'encoder/conv2d_9/BiasAdd/ReadVariableOp2P
&encoder/conv2d_9/Conv2D/ReadVariableOp&encoder/conv2d_9/Conv2D/ReadVariableOp2R
'encoder/dense_20/BiasAdd/ReadVariableOp'encoder/dense_20/BiasAdd/ReadVariableOp2P
&encoder/dense_20/MatMul/ReadVariableOp&encoder/dense_20/MatMul/ReadVariableOp2R
'encoder/dense_21/BiasAdd/ReadVariableOp'encoder/dense_21/BiasAdd/ReadVariableOp2P
&encoder/dense_21/MatMul/ReadVariableOp&encoder/dense_21/MatMul/ReadVariableOp2D
 encoder/z/BiasAdd/ReadVariableOp encoder/z/BiasAdd/ReadVariableOp2B
encoder/z/MatMul/ReadVariableOpencoder/z/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
(__inference_encoder_layer_call_fn_364096

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
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
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
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_363312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_364397

inputsA
'conv2d_8_conv2d_readvariableop_resource:6
(conv2d_8_biasadd_readvariableop_resource:A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource: B
(conv2d_10_conv2d_readvariableop_resource: @7
)conv2d_10_biasadd_readvariableop_resource:@B
(conv2d_11_conv2d_readvariableop_resource:@H7
)conv2d_11_biasadd_readvariableop_resource:H;
-batch_normalization_2_readvariableop_resource:H=
/batch_normalization_2_readvariableop_1_resource:HL
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:H;
'dense_20_matmul_readvariableop_resource:
?	?7
(dense_20_biasadd_readvariableop_resource:	?:
'dense_21_matmul_readvariableop_resource:	?H6
(dense_21_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??$batch_normalization_2/AssignNewValue?&batch_normalization_2/AssignNewValue_1?5batch_normalization_2/FusedBatchNormV3/ReadVariableOp?7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_2/ReadVariableOp?&batch_normalization_2/ReadVariableOp_1? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp?dense_20/BiasAdd/ReadVariableOp?dense_20/MatMul/ReadVariableOp?dense_21/BiasAdd/ReadVariableOp?dense_21/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? j
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@l
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????Hl
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H?
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:H*
dtype0?
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_11/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  ?
flatten_2/ReshapeReshape*batch_normalization_2/FusedBatchNormV3:y:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????	?
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype0?
dense_20/MatMulMatMulflatten_2/Reshape:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
dense_21/MatMulMatMuldense_20/Relu:activations:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H?
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0?
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hj
dense_21/SoftplusSoftplusdense_21/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Hy
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype0?
z/MatMulMatMuldense_20/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hv
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0|
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Hf
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: a
IdentityIdentityz/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Hp

Identity_1Identitydense_21/Softplus:activations:0^NoOp*
T0*'
_output_shapes
:?????????H?	
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_363353
input_7!
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
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_363312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_364432

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_21_layer_call_fn_364658

inputs
unknown:	?H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_363228o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_364749R
8conv2d_11_kernel_regularizer_abs_readvariableop_resource:@H
identity??/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOpg
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp8conv2d_11_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv2d_11_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: d
IdentityIdentity&conv2d_11/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364599

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_363584

inputs)
conv2d_8_363477:
conv2d_8_363479:)
conv2d_9_363482: 
conv2d_9_363484: *
conv2d_10_363487: @
conv2d_10_363489:@*
conv2d_11_363492:@H
conv2d_11_363494:H*
batch_normalization_2_363497:H*
batch_normalization_2_363499:H*
batch_normalization_2_363501:H*
batch_normalization_2_363503:H#
dense_20_363507:
?	?
dense_20_363509:	?"
dense_21_363512:	?H
dense_21_363514:H
z_363517:	?H
z_363519:H
identity

identity_1??-batch_normalization_2/StatefulPartitionedCall?!conv2d_10/StatefulPartitionedCall?/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_10/kernel/Regularizer/Square/ReadVariableOp?!conv2d_11/StatefulPartitionedCall?/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp?2conv2d_11/kernel/Regularizer/Square/ReadVariableOp? conv2d_8/StatefulPartitionedCall?.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOp? conv2d_9/StatefulPartitionedCall?.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_9/kernel/Regularizer/Square/ReadVariableOp? dense_20/StatefulPartitionedCall? dense_21/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_363477conv2d_8_363479*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_363081?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_363482conv2d_9_363484*
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
GPU2*0J 8? *M
fHRF
D__inference_conv2d_9_layer_call_and_return_conditional_losses_363113?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_363487conv2d_10_363489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_10_layer_call_and_return_conditional_losses_363145?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_363492conv2d_11_363494*
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
GPU2*0J 8? *N
fIRG
E__inference_conv2d_11_layer_call_and_return_conditional_losses_363177?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0batch_normalization_2_363497batch_normalization_2_363499batch_normalization_2_363501batch_normalization_2_363503*
Tin	
2*
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
GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_363037?
flatten_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_2_layer_call_and_return_conditional_losses_363198?
 dense_20/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0dense_20_363507dense_20_363509*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_363211?
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_363512dense_21_363514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_363228?
z/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0z_363517z_363519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_363244f
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_8_363477*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_8_363477*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_9_363482*&
_output_shapes
: *
dtype0?
conv2d_9/kernel/Regularizer/AbsAbs6conv2d_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_9/kernel/Regularizer/SumSum#conv2d_9/kernel/Regularizer/Abs:y:0,conv2d_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_9/kernel/Regularizer/mulMul*conv2d_9/kernel/Regularizer/mul/x:output:0(conv2d_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_9/kernel/Regularizer/addAddV2*conv2d_9/kernel/Regularizer/Const:output:0#conv2d_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_9_363482*&
_output_shapes
: *
dtype0?
"conv2d_9/kernel/Regularizer/SquareSquare9conv2d_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: |
#conv2d_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_9/kernel/Regularizer/Sum_1Sum&conv2d_9/kernel/Regularizer/Square:y:0,conv2d_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_9/kernel/Regularizer/mul_1Mul,conv2d_9/kernel/Regularizer/mul_1/x:output:0*conv2d_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_9/kernel/Regularizer/add_1AddV2#conv2d_9/kernel/Regularizer/add:z:0%conv2d_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_10_363487*&
_output_shapes
: @*
dtype0?
 conv2d_10/kernel/Regularizer/AbsAbs7conv2d_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_10/kernel/Regularizer/SumSum$conv2d_10/kernel/Regularizer/Abs:y:0-conv2d_10/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_10/kernel/Regularizer/mulMul+conv2d_10/kernel/Regularizer/mul/x:output:0)conv2d_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_10/kernel/Regularizer/addAddV2+conv2d_10/kernel/Regularizer/Const:output:0$conv2d_10/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_10/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_10_363487*&
_output_shapes
: @*
dtype0?
#conv2d_10/kernel/Regularizer/SquareSquare:conv2d_10/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @}
$conv2d_10/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_10/kernel/Regularizer/Sum_1Sum'conv2d_10/kernel/Regularizer/Square:y:0-conv2d_10/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_10/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_10/kernel/Regularizer/mul_1Mul-conv2d_10/kernel/Regularizer/mul_1/x:output:0+conv2d_10/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_10/kernel/Regularizer/add_1AddV2$conv2d_10/kernel/Regularizer/add:z:0&conv2d_10/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_11_363492*&
_output_shapes
:@H*
dtype0?
 conv2d_11/kernel/Regularizer/AbsAbs7conv2d_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
 conv2d_11/kernel/Regularizer/SumSum$conv2d_11/kernel/Regularizer/Abs:y:0-conv2d_11/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: g
"conv2d_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
 conv2d_11/kernel/Regularizer/mulMul+conv2d_11/kernel/Regularizer/mul/x:output:0)conv2d_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
 conv2d_11/kernel/Regularizer/addAddV2+conv2d_11/kernel/Regularizer/Const:output:0$conv2d_11/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
2conv2d_11/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_11_363492*&
_output_shapes
:@H*
dtype0?
#conv2d_11/kernel/Regularizer/SquareSquare:conv2d_11/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H}
$conv2d_11/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
"conv2d_11/kernel/Regularizer/Sum_1Sum'conv2d_11/kernel/Regularizer/Square:y:0-conv2d_11/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: i
$conv2d_11/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
"conv2d_11/kernel/Regularizer/mul_1Mul-conv2d_11/kernel/Regularizer/mul_1/x:output:0+conv2d_11/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
"conv2d_11/kernel/Regularizer/add_1AddV2$conv2d_11/kernel/Regularizer/add:z:0&conv2d_11/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: q
IdentityIdentity"z/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hz

Identity_1Identity)dense_21/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H?
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall"^conv2d_10/StatefulPartitionedCall0^conv2d_10/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_10/kernel/Regularizer/Square/ReadVariableOp"^conv2d_11/StatefulPartitionedCall0^conv2d_11/kernel/Regularizer/Abs/ReadVariableOp3^conv2d_11/kernel/Regularizer/Square/ReadVariableOp!^conv2d_8/StatefulPartitionedCall/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp!^conv2d_9/StatefulPartitionedCall/^conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_9/kernel/Regularizer/Square/ReadVariableOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall^z/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2b
/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp/conv2d_10/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2conv2d_10/kernel/Regularizer/Square/ReadVariableOp2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2b
/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp/conv2d_11/kernel/Regularizer/Abs/ReadVariableOp2h
2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2conv2d_11/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2`
.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp.conv2d_9/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_9/kernel/Regularizer/Square/ReadVariableOp1conv2d_9/kernel/Regularizer/Square/ReadVariableOp2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_z_layer_call_fn_364639

inputs
unknown:	?H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_363244o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_363993
input_7!
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
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_362984o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Hq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????H`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_7
?
?
__inference_loss_fn_0_364689Q
7conv2d_8_kernel_regularizer_abs_readvariableop_resource:
identity??.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_8/kernel/Regularizer/Square/ReadVariableOpf
!conv2d_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_8/kernel/Regularizer/AbsAbs6conv2d_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             ?
conv2d_8/kernel/Regularizer/SumSum#conv2d_8/kernel/Regularizer/Abs:y:0,conv2d_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: f
!conv2d_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
conv2d_8/kernel/Regularizer/mulMul*conv2d_8/kernel/Regularizer/mul/x:output:0(conv2d_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: ?
conv2d_8/kernel/Regularizer/addAddV2*conv2d_8/kernel/Regularizer/Const:output:0#conv2d_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: ?
1conv2d_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype0?
"conv2d_8/kernel/Regularizer/SquareSquare9conv2d_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:|
#conv2d_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             ?
!conv2d_8/kernel/Regularizer/Sum_1Sum&conv2d_8/kernel/Regularizer/Square:y:0,conv2d_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: h
#conv2d_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<?
!conv2d_8/kernel/Regularizer/mul_1Mul,conv2d_8/kernel/Regularizer/mul_1/x:output:0*conv2d_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: ?
!conv2d_8/kernel/Regularizer/add_1AddV2#conv2d_8/kernel/Regularizer/add:z:0%conv2d_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: c
IdentityIdentity%conv2d_8/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp/^conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_8/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp.conv2d_8/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_8/kernel/Regularizer/Square/ReadVariableOp1conv2d_8/kernel/Regularizer/Square/ReadVariableOp
?
?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364581

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????H?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_78
serving_default_input_7:0?????????<
dense_210
StatefulPartitionedCall:0?????????H5
z0
StatefulPartitionedCall:1?????????Htensorflow/serving/predict:??
?
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
?
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
?
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
?
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
?
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
?
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
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

Nkernel
Obias"
_tf_keras_layer
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses

Vkernel
Wbias"
_tf_keras_layer
?
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses

^kernel
_bias"
_tf_keras_layer
?
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
?
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
?
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
?
itrace_0
jtrace_1
ktrace_2
ltrace_32?
(__inference_encoder_layer_call_fn_363353
(__inference_encoder_layer_call_fn_364096
(__inference_encoder_layer_call_fn_364139
(__inference_encoder_layer_call_fn_363668?
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
 zitrace_0zjtrace_1zktrace_2zltrace_3
?
mtrace_0
ntrace_1
otrace_2
ptrace_32?
C__inference_encoder_layer_call_and_return_conditional_losses_364268
C__inference_encoder_layer_call_and_return_conditional_losses_364397
C__inference_encoder_layer_call_and_return_conditional_losses_363778
C__inference_encoder_layer_call_and_return_conditional_losses_363888?
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
 zmtrace_0zntrace_1zotrace_2zptrace_3
?B?
!__inference__wrapped_model_362984input_7"?
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
?
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
?
wtrace_02?
)__inference_conv2d_8_layer_call_fn_364406?
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
 zwtrace_0
?
xtrace_02?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_364432?
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
 zxtrace_0
):'2conv2d_8/kernel
:2conv2d_8/bias
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
?
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
?
~trace_02?
)__inference_conv2d_9_layer_call_fn_364441?
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
 z~trace_0
?
trace_02?
D__inference_conv2d_9_layer_call_and_return_conditional_losses_364467?
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
 ztrace_0
):' 2conv2d_9/kernel
: 2conv2d_9/bias
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_10_layer_call_fn_364476?
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_364502?
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
*:( @2conv2d_10/kernel
:@2conv2d_10/bias
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_11_layer_call_fn_364511?
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
E__inference_conv2d_11_layer_call_and_return_conditional_losses_364537?
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
*:(@H2conv2d_11/kernel
:H2conv2d_11/bias
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_2_layer_call_fn_364550
6__inference_batch_normalization_2_layer_call_fn_364563?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364581
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364599?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
):'H2batch_normalization_2/gamma
(:&H2batch_normalization_2/beta
1:/H (2!batch_normalization_2/moving_mean
5:3H (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_flatten_2_layer_call_fn_364604?
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_364610?
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_20_layer_call_fn_364619?
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
D__inference_dense_20_layer_call_and_return_conditional_losses_364630?
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
#:!
?	?2dense_20/kernel
:?2dense_20/bias
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
"__inference_z_layer_call_fn_364639?
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
=__inference_z_layer_call_and_return_conditional_losses_364649?
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
:	?H2z/kernel
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_21_layer_call_fn_364658?
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
D__inference_dense_21_layer_call_and_return_conditional_losses_364669?
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
": 	?H2dense_21/kernel
:H2dense_21/bias
?
?trace_02?
__inference_loss_fn_0_364689?
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
__inference_loss_fn_1_364709?
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
__inference_loss_fn_2_364729?
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
__inference_loss_fn_3_364749?
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
?B?
(__inference_encoder_layer_call_fn_363353input_7"?
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
(__inference_encoder_layer_call_fn_364096inputs"?
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
(__inference_encoder_layer_call_fn_364139inputs"?
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
(__inference_encoder_layer_call_fn_363668input_7"?
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
C__inference_encoder_layer_call_and_return_conditional_losses_364268inputs"?
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
C__inference_encoder_layer_call_and_return_conditional_losses_364397inputs"?
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
C__inference_encoder_layer_call_and_return_conditional_losses_363778input_7"?
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
C__inference_encoder_layer_call_and_return_conditional_losses_363888input_7"?
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
$__inference_signature_wrapper_363993input_7"?
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
`0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_8_layer_call_fn_364406inputs"?
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
D__inference_conv2d_8_layer_call_and_return_conditional_losses_364432inputs"?
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
a0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_9_layer_call_fn_364441inputs"?
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
D__inference_conv2d_9_layer_call_and_return_conditional_losses_364467inputs"?
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
b0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_conv2d_10_layer_call_fn_364476inputs"?
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
E__inference_conv2d_10_layer_call_and_return_conditional_losses_364502inputs"?
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
c0"
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_conv2d_11_layer_call_fn_364511inputs"?
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
E__inference_conv2d_11_layer_call_and_return_conditional_losses_364537inputs"?
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
?B?
6__inference_batch_normalization_2_layer_call_fn_364550inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_2_layer_call_fn_364563inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364581inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364599inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
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
*__inference_flatten_2_layer_call_fn_364604inputs"?
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
E__inference_flatten_2_layer_call_and_return_conditional_losses_364610inputs"?
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
)__inference_dense_20_layer_call_fn_364619inputs"?
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
D__inference_dense_20_layer_call_and_return_conditional_losses_364630inputs"?
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
"__inference_z_layer_call_fn_364639inputs"?
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
=__inference_z_layer_call_and_return_conditional_losses_364649inputs"?
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
)__inference_dense_21_layer_call_fn_364658inputs"?
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
D__inference_dense_21_layer_call_and_return_conditional_losses_364669inputs"?
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
__inference_loss_fn_0_364689"?
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
__inference_loss_fn_1_364709"?
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
__inference_loss_fn_2_364729"?
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
__inference_loss_fn_3_364749"?
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
!__inference__wrapped_model_362984?"#+,45>?@ANO^_VW8?5
.?+
)?&
input_7?????????
? "U?R
.
dense_21"?
dense_21?????????H
 
z?
z?????????H?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364581?>?@AM?J
C?@
:?7
inputs+???????????????????????????H
p 
? "??<
5?2
0+???????????????????????????H
? ?
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_364599?>?@AM?J
C?@
:?7
inputs+???????????????????????????H
p
? "??<
5?2
0+???????????????????????????H
? ?
6__inference_batch_normalization_2_layer_call_fn_364550?>?@AM?J
C?@
:?7
inputs+???????????????????????????H
p 
? "2?/+???????????????????????????H?
6__inference_batch_normalization_2_layer_call_fn_364563?>?@AM?J
C?@
:?7
inputs+???????????????????????????H
p
? "2?/+???????????????????????????H?
E__inference_conv2d_10_layer_call_and_return_conditional_losses_364502l+,7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2d_10_layer_call_fn_364476_+,7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv2d_11_layer_call_and_return_conditional_losses_364537l457?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????H
? ?
*__inference_conv2d_11_layer_call_fn_364511_457?4
-?*
(?%
inputs?????????@
? " ??????????H?
D__inference_conv2d_8_layer_call_and_return_conditional_losses_364432l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_8_layer_call_fn_364406_7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2d_9_layer_call_and_return_conditional_losses_364467l"#7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_9_layer_call_fn_364441_"#7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_dense_20_layer_call_and_return_conditional_losses_364630^NO0?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? ~
)__inference_dense_20_layer_call_fn_364619QNO0?-
&?#
!?
inputs??????????	
? "????????????
D__inference_dense_21_layer_call_and_return_conditional_losses_364669]^_0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? }
)__inference_dense_21_layer_call_fn_364658P^_0?-
&?#
!?
inputs??????????
? "??????????H?
C__inference_encoder_layer_call_and_return_conditional_losses_363778?"#+,45>?@ANO^_VW@?=
6?3
)?&
input_7?????????
p 

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_363888?"#+,45>?@ANO^_VW@?=
6?3
)?&
input_7?????????
p

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_364268?"#+,45>?@ANO^_VW??<
5?2
(?%
inputs?????????
p 

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_364397?"#+,45>?@ANO^_VW??<
5?2
(?%
inputs?????????
p

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
(__inference_encoder_layer_call_fn_363353?"#+,45>?@ANO^_VW@?=
6?3
)?&
input_7?????????
p 

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_363668?"#+,45>?@ANO^_VW@?=
6?3
)?&
input_7?????????
p

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_364096?"#+,45>?@ANO^_VW??<
5?2
(?%
inputs?????????
p 

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_364139?"#+,45>?@ANO^_VW??<
5?2
(?%
inputs?????????
p

 
? "=?:
?
0?????????H
?
1?????????H?
E__inference_flatten_2_layer_call_and_return_conditional_losses_364610a7?4
-?*
(?%
inputs?????????H
? "&?#
?
0??????????	
? ?
*__inference_flatten_2_layer_call_fn_364604T7?4
-?*
(?%
inputs?????????H
? "???????????	;
__inference_loss_fn_0_364689?

? 
? "? ;
__inference_loss_fn_1_364709"?

? 
? "? ;
__inference_loss_fn_2_364729+?

? 
? "? ;
__inference_loss_fn_3_3647494?

? 
? "? ?
$__inference_signature_wrapper_363993?"#+,45>?@ANO^_VWC?@
? 
9?6
4
input_7)?&
input_7?????????"U?R
.
dense_21"?
dense_21?????????H
 
z?
z?????????H?
=__inference_z_layer_call_and_return_conditional_losses_364649]VW0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? v
"__inference_z_layer_call_fn_364639PVW0?-
&?#
!?
inputs??????????
? "??????????H