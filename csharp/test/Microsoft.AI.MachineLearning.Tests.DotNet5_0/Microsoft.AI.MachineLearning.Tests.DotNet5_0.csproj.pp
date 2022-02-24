<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net5.0-windows10.0.17763.0</TargetFramework>
    <Platforms>x86;x64</Platforms>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.AI.DirectML" Version="1.8.0" />
    <PackageReference Include="Microsoft.AI.MachineLearning" Version="[PackageVersion]" />
  </ItemGroup>

  <ItemGroup>
    <None Include="..\..\testdata\squeezenet.onnx">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
    <None Include="..\..\..\winml\test\collateral\images\kitten_224.png">
      <CopyToOutputDirectory>Always</CopyToOutputDirectory>
      <Visible>true</Visible>
    </None>
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\Microsoft.AI.MachineLearning.Tests.Lib.DotNet5_0\Microsoft.AI.MachineLearning.Tests.Lib.DotNet5_0.csproj" />
  </ItemGroup>
</Project>
