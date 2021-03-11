#define LF_DETECTION
#ifdef LF_DETECTION
#include "arcsoft_liveface.h"
#pragma comment(lib, "libarcsoft_antisproof_liveface.lib")
#endif


//support one face
void main(char*IRData, char*Depthdata,  int IRWidth, int IRHeight, int DepthWidth, int DepthHeight, ALT_FACE_INFOR* outline)
{

#ifdef LF_DETECTION
	MVoid*	LFMemBuffer;
	MHandle LFMemHandle;
	ASVLOFFSCREEN IRImageInfo;
	ASVLOFFSCREEN DepthDataImageInfo;
	ASAE_ENGINE LFEngine;
	ASAE_RACERESULT LFResult;
	ASAE_FACEOUTLINE_INPUT face_landmark;

//initial	
	LFEngine = 0;
	LFMemBuffer = malloc(1024*1024*50);
	LFMemHandle = MMemMgrCreate(LFMemBuffer,1024*1024*50);
	int res = ASAE_InitLFEngine(LFMemHandle, &LFEngine);
	if (res != MOK)
	{
		printf("%d\n",res);
	 }
	 
	FILE * fp = fopen("RGB_IR_MODEL.data", "rb");
	size_t read_size = 0;
	fseek(fp, 0, SEEK_END);
	read_size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char* cnn_model = (char *)malloc((read_size)* sizeof(char));
	fread(cnn_model, read_size, sizeof(char), fp);
	fclose(fp);

	res = ASLF_SetBGRImage(LFMemHandle, LFEngine, cnn_model + ((int*)cnn_model)[0]);
	res = ASLF_SetIRImage(LFMemHandle, LFEngine, cnn_model);	
	PThreshold.thresholdmodel_BGR = 0.6;
	PThreshold.thresholdmodel_IR = 0.65;

	while(1)
	{
		IRImageInfo.i32Width = IRWidth;
		IRImageInfo.i32Height = IRHeight;
		IRImageInfo.ppu8Plane[0] = IRData;
		IRImageInfo.pi32Pitch[0] = IRImageInfo.i32Width;
		IRImageInfo.u32PixelArrayFormat = ASVL_PAF_DEPTH_U16;
	
	
		face_landmark.b_detect = outline->b_detect;
		face_landmark.nFace = outline->nFace;
		face_landmark.pitch_angle = outline->pitch_angle;
		face_landmark.roll_angle = outline->roll_angle;
		face_landmark.yaw_angle = outline->yaw_angle;
		face_landmark.rcFace = outline->rcFace;
		face_landmark.landmarks = (MFPOINT*)outline->landmarks_122;
		//preview 
		//int res = ASAE_LiveFace_Preview(LFMemHandle, LFEngine, NULL, &IRImageInfo, NULL, &PThreshold, &face_landmark, &LFResult, 0, 0);
		//static image
		int res = ASAE_LiveFace_StaticImage(LFMemHandle, LFEngine, NULL, &IRImageInfo, NULL, &PThreshold, &face_landmark, &LFResult, 0, 0);
		if (res != MOK)
		{
			printf("detect error!\n");
			return;
		}
	}



	// result	
	if(LFResult.pLFResultArray[0]==0)
		Fate++;
	else if(LFResult.pLFResultArray[0]==1)
		Live++;
	else 
	   	unknown++;
	//Printf(¡°BGR_live:%.4f¡±, LFResult. cl_conf[0]); 
	Printf(¡°IR_live:%.4f¡±, LFResult. cl_conf[1]); 


//release

	if (LFEngine)
	{
		ASAE_UninitLFEngine(LFMemHandle, &LFEngine);
	}
	if (LFMemHandle)
	{
		MMemMgrDestroy(LFMemHandle);
		LFMemHandle = 0;
	}
	if (LFMemBuffer)
	{
		free(LFMemBuffer);
		LFMemBuffer = NULL;
	}
#endif
}







