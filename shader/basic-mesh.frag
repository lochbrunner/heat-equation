#version 130

//uniform sampler2D texture0;
uniform sampler1D texture1;

in vec4 color;
in vec2 texCoord0;

//in vec3 lightDir;
//in vec3 normalDir;
//in vec3 biNormalDir;
//in vec3 tangentDir;


out vec4 outColor;

void main(void) {
	outColor = color.y*texture(texture1, color.x);
	//outColor = color.y*vec4(1.0, 1.0, 1.0, 1.0); //Nur um bumpmapping zu testen
	outColor.a = 1.0;
}
