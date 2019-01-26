#version 130


uniform sampler2D texture0;

uniform mat4 projection_matrix;
uniform mat4 modelview_matrix;
uniform mat3 normal_matrix;



struct light {
	vec4 position;
	vec4 diffuse;
};

uniform light light0;

in vec4 a_Vertex;
in vec2 a_TexCoord0;
in vec4 a_Normal;
in vec4 a_BiNormal;


out vec4 color;
out vec2 texCoord0;

void main(void) 
{
	texCoord0 = a_TexCoord0;
	vec3 normalDir;
	vec3 lightDir;
	vec3 biNormalDir;
	vec3 tangentDir;
	float isBorder = 0.5;

	vec4 tvalue = texture2DLod(texture0, a_TexCoord0.st, 0.0);

	vec4 vertex = a_Vertex;
	vertex.y = tvalue.x;
	
	vec4 pos = modelview_matrix * vertex;
	normalDir = normalize(normal_matrix * a_Normal.xyz);
	biNormalDir = normalize(normal_matrix * a_BiNormal.xyz);
	tangentDir=cross(normalDir,biNormalDir);

	vec3 newNormal = normalize(normalDir+biNormalDir*tvalue.z-tangentDir*tvalue.w);
	
	vec3 lightPos = light0.position.xyz;
	
	lightDir = normalize(lightPos - pos.xyz);
	float scale = (0.2+0.8*abs(dot(newNormal, lightDir)));
	color = vec4(tvalue.x, scale, isBorder, 1.0);
	gl_Position = projection_matrix * pos;
}
