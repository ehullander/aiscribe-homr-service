// Disposable CPU benchmark only. Never bind live service credentials or models.
targetScope = 'resourceGroup'
param location string
param name string
param dnsLabel string
param runtimeImage string
param bundleSha256 string
param sourceUrl string
param bootstrap string
@secure()
param bundleUrl string
@secure()
param benchmarkToken string
@secure()
param tlsCertificate string
@secure()
param tlsPrivateKey string
resource benchmark 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: name
  location: location
  properties: {
    osType: 'Linux'
    restartPolicy: 'Never'
    containers: [{
      name: 'homr-cpu-benchmark'
      properties: {
        image: runtimeImage
        command: ['python', '-u', '-c', bootstrap]
        resources: { requests: { cpu: 4, memoryInGB: 8 } }
        ports: [{ port: 8443, protocol: 'TCP' }]
        environmentVariables: [
          { name: 'HOMR_BUNDLE_URL', secureValue: bundleUrl }
          { name: 'HOMR_BUNDLE_SHA256', value: bundleSha256 }
          { name: 'HOMR_BENCHMARK_TOKEN', secureValue: benchmarkToken }
          { name: 'HOMR_TLS_CERT', secureValue: tlsCertificate }
          { name: 'HOMR_TLS_KEY', secureValue: tlsPrivateKey }
          { name: 'HOMR_SOURCE_URL', value: sourceUrl }
        ]
      }
    }]
    ipAddress: {
      type: 'Public'
      dnsNameLabel: dnsLabel
      ports: [{ port: 8443, protocol: 'TCP' }]
    }
  }
}
output hostname string = benchmark.properties.ipAddress.fqdn
