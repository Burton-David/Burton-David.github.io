<xsl:stylesheet version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
>

<xsl:variable name="n" select="2"/>

<xsl:template match="doc">
<xsl:value-of select="item[$n]"/>
</xsl:template>

</xsl:stylesheet>
