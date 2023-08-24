// shadowmap.vert
#version 450

layout (location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform lightMatrix
{
	mat4 lightViewProjection;
} uScene;

void main()
{
	vec4 lightSpace = uScene.lightViewProjection * vec4(position, 1.0f);
	gl_Position = vec4(lightSpace.x,lightSpace.y,lightSpace.z + 0.0458f,lightSpace.w);
}
