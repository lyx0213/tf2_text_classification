эг
—£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878ФШ
Ъ
fast_text/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
„Шd*/
shared_name fast_text/embedding/embeddings
У
2fast_text/embedding/embeddings/Read/ReadVariableOpReadVariableOpfast_text/embedding/embeddings* 
_output_shapes
:
„Шd*
dtype0
И
fast_text/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_namefast_text/dense/kernel
Б
*fast_text/dense/kernel/Read/ReadVariableOpReadVariableOpfast_text/dense/kernel*
_output_shapes

:d*
dtype0
А
fast_text/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namefast_text/dense/bias
y
(fast_text/dense/bias/Read/ReadVariableOpReadVariableOpfast_text/dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Џ

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Х

valueЛ
BИ
 BБ

~
	embedding
softmax
regularization_losses
	variables
trainable_variables
	keras_api

signatures
b

embeddings
	regularization_losses

	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 

0
1
2

0
1
2
≠
metrics

layers
regularization_losses
	variables
layer_metrics
layer_regularization_losses
trainable_variables
non_trainable_variables
 
ca
VARIABLE_VALUEfast_text/embedding/embeddings/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
≠
metrics

layers
	regularization_losses

	variables
layer_metrics
layer_regularization_losses
trainable_variables
non_trainable_variables
US
VARIABLE_VALUEfast_text/dense/kernel)softmax/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEfast_text/dense/bias'softmax/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
metrics

layers
regularization_losses
	variables
layer_metrics
 layer_regularization_losses
trainable_variables
!non_trainable_variables
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Е
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1fast_text/embedding/embeddingsfast_text/dense/kernelfast_text/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_10181
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
І
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2fast_text/embedding/embeddings/Read/ReadVariableOp*fast_text/dense/kernel/Read/ReadVariableOp(fast_text/dense/bias/Read/ReadVariableOpConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_10365
ж
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefast_text/embedding/embeddingsfast_text/dense/kernelfast_text/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_10384цы
ш
л
 __inference__wrapped_model_10048
input_1@
<fast_text_embedding_embedding_lookup_readvariableop_resource2
.fast_text_dense_matmul_readvariableop_resource3
/fast_text_dense_biasadd_readvariableop_resource
identityИй
3fast_text/embedding/embedding_lookup/ReadVariableOpReadVariableOp<fast_text_embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype025
3fast_text/embedding/embedding_lookup/ReadVariableOpа
)fast_text/embedding/embedding_lookup/axisConst*F
_class<
:8loc:@fast_text/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2+
)fast_text/embedding/embedding_lookup/axisу
$fast_text/embedding/embedding_lookupGatherV2;fast_text/embedding/embedding_lookup/ReadVariableOp:value:0input_12fast_text/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*F
_class<
:8loc:@fast_text/embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2&
$fast_text/embedding/embedding_lookupѕ
-fast_text/embedding/embedding_lookup/IdentityIdentity-fast_text/embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2/
-fast_text/embedding/embedding_lookup/IdentityЖ
 fast_text/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 fast_text/Mean/reduction_indicesљ
fast_text/MeanMean6fast_text/embedding/embedding_lookup/Identity:output:0)fast_text/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
fast_text/Meanљ
%fast_text/dense/MatMul/ReadVariableOpReadVariableOp.fast_text_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02'
%fast_text/dense/MatMul/ReadVariableOpі
fast_text/dense/MatMulMatMulfast_text/Mean:output:0-fast_text/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fast_text/dense/MatMulЉ
&fast_text/dense/BiasAdd/ReadVariableOpReadVariableOp/fast_text_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&fast_text/dense/BiasAdd/ReadVariableOpЅ
fast_text/dense/BiasAddBiasAdd fast_text/dense/MatMul:product:0.fast_text/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
fast_text/dense/BiasAddС
fast_text/dense/SoftmaxSoftmax fast_text/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
fast_text/dense/Softmaxu
IdentityIdentity!fast_text/dense/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
А
с
D__inference_fast_text_layer_call_and_return_conditional_losses_10199
input_16
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЋ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02+
)embedding/embedding_lookup/ReadVariableOp¬
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisЅ
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0input_1(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding/embedding_lookup±
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/Identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesХ
MeanMean,embedding/embedding_lookup/Identity:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense/MatMul/ReadVariableOpМ
dense/MatMulMatMulMean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
э
Н
)__inference_fast_text_layer_call_fn_10286

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_fast_text_layer_call_and_return_conditional_losses_101342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
э
р
D__inference_fast_text_layer_call_and_return_conditional_losses_10257

inputs6
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЋ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02+
)embedding/embedding_lookup/ReadVariableOp¬
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisј
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0inputs(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding/embedding_lookup±
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/Identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesХ
MeanMean,embedding/embedding_lookup/Identity:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense/MatMul/ReadVariableOpМ
dense/MatMulMatMulMean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
®
@__inference_dense_layer_call_and_return_conditional_losses_10086

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d:::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
э
Н
)__inference_fast_text_layer_call_fn_10297

inputs
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_fast_text_layer_call_and_return_conditional_losses_101592
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ш
Х
!__inference__traced_restore_10384
file_prefix3
/assignvariableop_fast_text_embedding_embeddings-
)assignvariableop_1_fast_text_dense_kernel+
'assignvariableop_2_fast_text_dense_bias

identity_4ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ђ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Є
valueЃBЂB/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)softmax/kernel/.ATTRIBUTES/VARIABLE_VALUEB'softmax/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slicesњ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЃ
AssignVariableOpAssignVariableOp/assignvariableop_fast_text_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOp)assignvariableop_1_fast_text_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ђ
AssignVariableOp_2AssignVariableOp'assignvariableop_2_fast_text_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp•

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_3Ч

Identity_4IdentityIdentity_3:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
T0*
_output_shapes
: 2

Identity_4"!

identity_4Identity_4:output:0*!
_input_shapes
: :::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
÷
И
#__inference_signature_wrapper_10181
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_100482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
А
О
)__inference_fast_text_layer_call_fn_10228
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_fast_text_layer_call_and_return_conditional_losses_101342
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
о
ё
D__inference_fast_text_layer_call_and_return_conditional_losses_10134

inputs
embedding_10123
dense_10128
dense_10130
identityИҐdense/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallЗ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_10123*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_100612#
!embedding/StatefulPartitionedCallr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesУ
MeanMean*embedding/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЙ
dense/StatefulPartitionedCallStatefulPartitionedCallMean:output:0dense_10128dense_10130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_100862
dense/StatefulPartitionedCallЊ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А
О
)__inference_fast_text_layer_call_fn_10239
input_1
unknown
	unknown_0
	unknown_1
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_fast_text_layer_call_and_return_conditional_losses_101592
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
™	
С
D__inference_embedding_layer_call_and_return_conditional_losses_10061

inputs,
(embedding_lookup_readvariableop_resource
identityИ≠
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02!
embedding_lookup/ReadVariableOp§
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axisО
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding_lookupУ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identityz
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒ
o
)__inference_embedding_layer_call_fn_10313

inputs
unknown
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_100612
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А
с
D__inference_fast_text_layer_call_and_return_conditional_losses_10217
input_16
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЋ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02+
)embedding/embedding_lookup/ReadVariableOp¬
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisЅ
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0input_1(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding/embedding_lookup±
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/Identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesХ
MeanMean,embedding/embedding_lookup/Identity:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense/MatMul/ReadVariableOpМ
dense/MatMulMatMulMean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
о
ё
D__inference_fast_text_layer_call_and_return_conditional_losses_10159

inputs
embedding_10148
dense_10153
dense_10155
identityИҐdense/StatefulPartitionedCallҐ!embedding/StatefulPartitionedCallЗ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_10148*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€d*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_100612#
!embedding/StatefulPartitionedCallr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesУ
MeanMean*embedding/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЙ
dense/StatefulPartitionedCallStatefulPartitionedCallMean:output:0dense_10153dense_10155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_100862
dense/StatefulPartitionedCallЊ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
э
р
D__inference_fast_text_layer_call_and_return_conditional_losses_10275

inputs6
2embedding_embedding_lookup_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЋ
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02+
)embedding/embedding_lookup/ReadVariableOp¬
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axisј
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0inputs(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding/embedding_lookup±
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2%
#embedding/embedding_lookup/Identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesХ
MeanMean,embedding/embedding_lookup/Identity:output:0Mean/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€d2
MeanЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense/MatMul/ReadVariableOpМ
dense/MatMulMatMulMean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*2
_input_shapes!
:€€€€€€€€€::::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘
z
%__inference_dense_layer_call_fn_10333

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_100862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs
™	
С
D__inference_embedding_layer_call_and_return_conditional_losses_10306

inputs,
(embedding_lookup_readvariableop_resource
identityИ≠
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
„Шd*
dtype02!
embedding_lookup/ReadVariableOp§
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axisО
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*+
_output_shapes
:€€€€€€€€€d2
embedding_lookupУ
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2
embedding_lookup/Identityz
IdentityIdentity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:€€€€€€€€€d2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ё
Ц
__inference__traced_save_10365
file_prefix=
9savev2_fast_text_embedding_embeddings_read_readvariableop5
1savev2_fast_text_dense_kernel_read_readvariableop3
/savev2_fast_text_dense_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8ded69a42f2246e58286f258685bc140/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¶
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Є
valueЃBЂB/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)softmax/kernel/.ATTRIBUTES/VARIABLE_VALUEB'softmax/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices№
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_fast_text_embedding_embeddings_read_readvariableop1savev2_fast_text_dense_kernel_read_readvariableop/savev2_fast_text_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 : :
„Шd:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
„Шd:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
≠
®
@__inference_dense_layer_call_and_return_conditional_losses_10324

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€d:::O K
'
_output_shapes
:€€€€€€€€€d
 
_user_specified_nameinputs"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:±D
“
	embedding
softmax
regularization_losses
	variables
trainable_variables
	keras_api

signatures
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature"ъ
_tf_keras_modelа{"class_name": "FastText", "name": "fast_text", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "FastText"}}
¶

embeddings
	regularization_losses

	variables
trainable_variables
	keras_api
%__call__
*&&call_and_return_all_conditional_losses"З
_tf_keras_layerн{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 15]}, "dtype": "float32", "input_dim": 134231, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 15}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 15]}}
р

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
'__call__
*(&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 15, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 100]}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 
metrics

layers
regularization_losses
	variables
layer_metrics
layer_regularization_losses
trainable_variables
non_trainable_variables
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
,
)serving_default"
signature_map
2:0
„Шd2fast_text/embedding/embeddings
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
metrics

layers
	regularization_losses

	variables
layer_metrics
layer_regularization_losses
trainable_variables
non_trainable_variables
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
(:&d2fast_text/dense/kernel
": 2fast_text/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
≠
metrics

layers
regularization_losses
	variables
layer_metrics
 layer_regularization_losses
trainable_variables
!non_trainable_variables
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
т2п
)__inference_fast_text_layer_call_fn_10228
)__inference_fast_text_layer_call_fn_10239
)__inference_fast_text_layer_call_fn_10297
)__inference_fast_text_layer_call_fn_10286ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ё2џ
D__inference_fast_text_layer_call_and_return_conditional_losses_10257
D__inference_fast_text_layer_call_and_return_conditional_losses_10217
D__inference_fast_text_layer_call_and_return_conditional_losses_10275
D__inference_fast_text_layer_call_and_return_conditional_losses_10199ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ё2џ
 __inference__wrapped_model_10048ґ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
”2–
)__inference_embedding_layer_call_fn_10313Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_embedding_layer_call_and_return_conditional_losses_10306Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_dense_layer_call_fn_10333Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_dense_layer_call_and_return_conditional_losses_10324Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
2B0
#__inference_signature_wrapper_10181input_1Р
 __inference__wrapped_model_10048l0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€†
@__inference_dense_layer_call_and_return_conditional_losses_10324\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "%Ґ"
К
0€€€€€€€€€
Ъ x
%__inference_dense_layer_call_fn_10333O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€d
™ "К€€€€€€€€€І
D__inference_embedding_layer_call_and_return_conditional_losses_10306_/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€d
Ъ 
)__inference_embedding_layer_call_fn_10313R/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€dЃ
D__inference_fast_text_layer_call_and_return_conditional_losses_10199f8Ґ5
.Ґ+
!К
input_1€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ѓ
D__inference_fast_text_layer_call_and_return_conditional_losses_10217f8Ґ5
.Ґ+
!К
input_1€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≠
D__inference_fast_text_layer_call_and_return_conditional_losses_10257e7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≠
D__inference_fast_text_layer_call_and_return_conditional_losses_10275e7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ж
)__inference_fast_text_layer_call_fn_10228Y8Ґ5
.Ґ+
!К
input_1€€€€€€€€€
p

 
™ "К€€€€€€€€€Ж
)__inference_fast_text_layer_call_fn_10239Y8Ґ5
.Ґ+
!К
input_1€€€€€€€€€
p 

 
™ "К€€€€€€€€€Е
)__inference_fast_text_layer_call_fn_10286X7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€Е
)__inference_fast_text_layer_call_fn_10297X7Ґ4
-Ґ*
 К
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ю
#__inference_signature_wrapper_10181w;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€