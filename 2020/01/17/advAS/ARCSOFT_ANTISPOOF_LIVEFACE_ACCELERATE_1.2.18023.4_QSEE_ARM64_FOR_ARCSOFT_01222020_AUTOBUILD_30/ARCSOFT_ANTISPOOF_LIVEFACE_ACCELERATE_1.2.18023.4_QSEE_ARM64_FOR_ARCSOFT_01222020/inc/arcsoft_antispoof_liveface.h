/*******************************************************************************
Copyright(c) ArcSoft, All right reserved.

This file is ArcSoft's property. It contains ArcSoft's trade secret, proprietary 
and confidential information. 

The information and code contained in this file is only for authorized ArcSoft 
employees to design, create, modify, or review.

DO NOT DISTRIBUTE, DO NOT DUPLICATE OR TRANSMIT IN ANY FORM WITHOUT PROPER 
AUTHORIZATION.

If you are not an intended recipient of this file, you must not copy, 
distribute, modify, or take any action in reliance on it. 

If you have received this file in error, please immediately notify ArcSoft and 
permanently delete the original and any copy of any file and any printout 
thereof.
*******************************************************************************/

#ifndef _ASLF_ANTISPOOF_LIVEFACE_
#define _ASLF_ANTISPOOF_LIVEFACE_

#include "amcomdef.h"
#include "merror.h"
#include "asvloffscreen.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef MHandle AFA_LFEngine;
typedef MHandle AFA_ArcneatCallbacks;
typedef MHandle AFA_ArcneatHeapParam;

/*face orientation code*/
enum ASLF_OrientCode{
	ASLF_FOC_0 = 0x1,		/* 0 degree */
	ASLF_FOC_90 = 0x2,		/* 90 degree */
	ASLF_FOC_270 = 0x3,		/* 270 degree */
	ASLF_FOC_180 = 0x4,      /* 180 degree */
	ASLF_FOC_30 = 0x5,		/* 30 degree */
	ASLF_FOC_60 = 0x6,		/* 60 degree */
	ASLF_FOC_120 = 0x7,		/* 120 degree */
	ASLF_FOC_150 = 0x8,		/* 150 degree */
	ASLF_FOC_210 = 0x9,		/* 210 degree */
	ASLF_FOC_240 = 0xa,		/* 240 degree */
	ASLF_FOC_300 = 0xb,		/* 300 degree */
	ASLF_FOC_330 = 0xc		/* 330 degree */
};

typedef struct{
	MRECT		* rcFace;
	MInt32		   nFace;
	MPOINTF		* landmarks;
	MFloat		* roll_angle;
	MFloat		* yaw_angle;
	MFloat		* pitch_angle;
}ASLF_FACEOUTLINE_INPUT, *LPASLF_FACEOUTLINE_INPUT;

typedef struct{
	MFloat		thresholdmodel_BGR;
	MFloat		thresholdmodel_IR;
	MFloat		thresholdmodel_DEPTH;
    MFloat		thresholdmodel_IR_DEPTH;
}ASLF_THRESHOLDPARM, *LPASLF_THRESHOLDPARM;

/***************************************************/
//pLFResult
//-1: Initial state
//-2: Number of faces != 1
//-3: too small width of face
//-4£ºtoo large face of angle
//-5: out of boundary face
//-6: error depth 
//-7: too bright IR_Img
//-8: too dark IR_Img
//¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤¡¤
//-11£ºhighlight_Value / rect >0.3
//-12: gray_value > 230
//-13: dark_Value / rect >0.8
//-14: gray_value < 30
/***************************************************/

typedef struct{
	MInt32 			pLFResult;			/*The LF result. Only support one face*/
	MFloat			IR_conf;           /*[output] the confidence of IR */
	MFloat			BGR_conf;           /*[output] the confidence of BGR */
	MFloat			Depth_conf;           /*[output] the confidence of Depth */
	MFloat			IR_conf_sequence;           /*[output] the confidence of IR preview */
	MFloat			BGR_conf_sequence;           /*[output] the confidence of BGR preview */
	MFloat			Depth_conf_sequence;           /*[output] the confidence of Depth preview */

}ASLF_LFRESULT, *LPASLF_LFRESULT;


typedef struct
{
	MInt32 lCodebase;     /* code base version number */
	MInt32 lMajor;        /* major version number */
	MInt32 lMinor;        /* minor version number */
	MInt32 lBuild;        /* build version number, increasable only */
	MTChar Version[50]; 	/* version in string form */
	MTChar BuildDate[20];	/* latest build Date */
	MTChar CopyRight[80];	/* copyright */
} ArcSoft_Antispoof_LiveFace_Version;

/************************************************************************
The function used to get obj_heap_size, param_heap_size and noload_heap_size.
orient_flag      [in]  define the priority of face orientation
detect_scale     [in]  define detect face size
obj_heap_size    [out] obj_heap_size
param_heap_size  [out] param_heap_size
noload_heap_size [out] noload_heap_size
************************************************************************/
#define AFA_MAX_ARCNET_NUM 2
typedef struct {
  int obj_heap_size;
  int param_heap_size;
  int noload_heap_size;
  int input_num;
  int output_num;
  int input_size[AFA_MAX_ARCNET_NUM];
  int output_size[AFA_MAX_ARCNET_NUM];
} AFA_Arcneat_Buffer_Size_Param;

MRESULT AFA_GetArcneatBufferSize(AFA_Arcneat_Buffer_Size_Param *arcneat_buffer_size_param);

/************************************************************************
The function used to Initialize the face detection engine.
hMemMgr       [in]  user defined memory manager
pEngine     [out] pointer to the LFEngine
model_data   [in]  model data, can be null
************************************************************************/

typedef struct {
  int num;
  AFA_ArcneatCallbacks arcneat_callbacks_handle[AFA_MAX_ARCNET_NUM];
  AFA_ArcneatHeapParam arcneat_heap_param_handle[AFA_MAX_ARCNET_NUM];
} AFA_Arcneat_Init_Param;

MInt32 ASLF_InitLFEngine(
	MHandle 	hMemMgr, 
	AFA_LFEngine  *pEngine);



MInt32 ASLF_SetBGRImage( 	/* Return LiveFace engine handle. If MNull, meaning bad model*/
	MHandle hMemMgr,		/* [in]  The handle of memory manager*/
	AFA_LFEngine pEngine, 	/* [out]  Pointer pointing to an LF engine*/
	AFA_Arcneat_Init_Param *arcneat_init_param, /*[in] init neat param*/
	MVoid* modeldata		/* [in]  IR model data,can set NULL, if modeldata != NULL, load external modeldata£¬*/
	);


MRESULT AFA_PreProcess(MHandle hMemMgr,
                        AFA_LFEngine pEngine, 
						LPASLF_FACEOUTLINE_INPUT	pLandmarkRes,
						LPASVLOFFSCREEN				pBGRImginfo,
                        AFA_ArcneatHeapParam arcneat_heap_param
                        );
							   

MRESULT AFA_PostProcess(MHandle hMemMgr,
  AFA_LFEngine pEngine,
  LPASLF_THRESHOLDPARM		PThreshold,     /* [in]  The threshold of two models*/
  AFA_ArcneatHeapParam arcneat_heap_param,
  LPASLF_LFRESULT pLFRes);

/************************************************************************
 * The function used to release the LiveFace module. 
 ************************************************************************/
MInt32 ASLF_UninitLFEngine(
   	MHandle            hMemMgr,		/* [in]  The handle of memory manager*/
	AFA_LFEngine 	   *pEngine		/* [in]  Pointer pointing to an ASBD_ENGINE structure containing the data of LiveFace engine*/
);


/************************************************************************
 * The function used to get version information of Liveface library. 
 ************************************************************************/
const ArcSoft_Antispoof_LiveFace_Version * ArcSoft_LiveFace_GetVersion();


#ifdef __cplusplus
}
#endif

#endif
