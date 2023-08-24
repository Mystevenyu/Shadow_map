#version 450

layout (location = 0) in vec2 v2fTexCoord;
layout (location = 1) in vec3 oNormal;
layout (location = 2) in vec3 oPosition;
layout (location = 3) in vec3 oCameraPos;
layout (location = 4) in vec3 oLightPos;
layout (location = 5) in vec3 oLightColor;


layout(set = 1,binding = 0) uniform sampler2D unm1;//baseColor
layout(set = 1,binding = 1) uniform sampler2D unm2;
layout(set = 1,binding = 2) uniform sampler2D unm3;

layout(set = 2, binding = 0) uniform sampler2DShadow shadowMap;

layout(set = 3, binding = 0) uniform lightMatrix
{
	mat4 lightViewProjection;
}uLightMa;
layout (location = 0) out vec4 outColor;

float M_PI = 3.1415926;


float BlinnPhongDist(float shininess,vec3 surfaceNor,vec3 halfVec)
{
	float surNDotHalfV = max(dot(surfaceNor,halfVec),0.0);
	return ((shininess + 2) / (2 * M_PI)) * pow(surNDotHalfV,shininess);
}

float CookTorranceModel(vec3 surfaceNor,vec3 halfVec,vec3 lightDir,vec3 viewDir)
{
	float nhnv = 2 * (max(dot(surfaceNor,halfVec),0.f) * max(dot(surfaceNor,viewDir),0.f)) / (dot(viewDir,halfVec) + 0.0001);
	float nhnl = 2 * (max(dot(surfaceNor,halfVec),0.f) * max(dot(surfaceNor,lightDir),0.f)) / (dot(viewDir,halfVec) + 0.0001);
	return min(1.0,min(nhnv,nhnl));
}

vec3 sRGBtoLinear(vec3 sRGB) {
    return pow(sRGB, vec3(2.2));
}


void main()
{
	vec3 L0 = vec3(0, 0, 0);

	vec3 lightDir = normalize(oLightPos - oPosition);

	//1.2
	vec3 materBaseColor =texture(unm1,v2fTexCoord).rgb;
	float alpha = texture(unm1,v2fTexCoord).a;
	float roughness = texture(unm2,v2fTexCoord).r;
	float materMetalness = texture(unm3,v2fTexCoord).r;


	//view direction, half vector
	vec3 V = normalize(oCameraPos - oPosition);
	vec3 H = normalize(V + lightDir);
	vec3 N = normalize(oNormal);

	float shininess = (2.f / (pow(roughness, 4) + 0.0001)) - 2.f;

	vec3 F0 = ((1 - materMetalness) * vec3(0.04f)) + materMetalness * materBaseColor; 

	//Ldiffuse¡¢G¡¢D¡¢F
	vec3 F = F0 + (1.f - F0) * pow((1.f - dot(H,V)), 5);    //Fresnel schlick approximation
	float D = BlinnPhongDist(shininess,N,H);
	float G = CookTorranceModel(N,H,lightDir,V);
	vec3 Ldiffuse =  (materBaseColor / M_PI) * (vec3(1.f) - F) * (1.f - materMetalness);

	//BRDF
	vec3 up = D * F * G;
	float down = 4.0 * max(dot(N,V),0) * max(dot(N,lightDir),0) + 0.0001;
	vec3 BRDF = Ldiffuse + up / down;


	vec3 Lambient = 0.04f * materBaseColor;

	float Clight = max(dot(N,lightDir),0);
	vec4 transLightSpace = uLightMa.lightViewProjection * vec4(oPosition,1.f);
	vec3 projCoord = transLightSpace.xyz / transLightSpace.w;
	projCoord = projCoord * 0.5 + 0.5;

	// PCF parameters
const int PCF_SIZE = 4;   // size of PCF filter kernel (should be odd)
const float PCF_SPREAD = 0.001f;   // distance between PCF samples in texture space

// . . . (code omitted)

//	float shadow = 0.0f;
//	for (int i = -PCF_SIZE/2; i <= PCF_SIZE/2; ++i)
//	{
//		for (int j = -PCF_SIZE/2; j <= PCF_SIZE/2; ++j)
//		{
//			vec3 offset = vec3(float(i), float(j), 0.0f) * PCF_SPREAD;
//			vec4 shadowMapCoords = vec4(projCoord.xy + offset.xy, projCoord.z, 1.0f);
//			shadow += textureProj(shadowMap, shadowMapCoords);
//		}
//	}
//// average shadow samples
//	shadow /= float(PCF_SIZE * PCF_SIZE);
	
	float shadow = textureProj(shadowMap, vec4(projCoord,1.f));
	//float shadow = textureProj(shadowMap, vec4(projCoord - 0.001 * 2.5f,1.0f));

	Clight *= shadow;
	L0 = L0 + (Lambient + BRDF  * oLightColor * Clight);
	if(texture(unm1,v2fTexCoord).a < 0.5f) discard;	

	outColor = vec4(L0,1.f);
}

