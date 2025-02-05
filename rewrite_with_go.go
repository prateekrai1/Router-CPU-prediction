package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"

	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcap"
	"github.com/google/gopacket/pcapgo"
)

// Define new IP and MAC addresses
const (
	newSourceIP      = "192.168.1.183"    // Change this to the desired new source IP
	newDestinationIP = "192.168.192.130"  // Change this to the desired new destination IP
	routerMAC        = "94:83:C4:11:10:F7" // Change this to the router's MAC address
)

func rewritePcap(inputFile, outputFile string) error {
	// Open the original pcap file
	handle, err := pcap.OpenOffline(inputFile)
	if err != nil {
		return fmt.Errorf("error opening pcap file: %v", err)
	}
	defer handle.Close()

	// Create a new pcap file for writing
	outFile, err := os.Create(outputFile)
	if err != nil {
		return fmt.Errorf("error creating output pcap file: %v", err)
	}
	defer outFile.Close()

	writer := pcapgo.NewWriter(outFile)
	writer.WriteFileHeader(65536, handle.LinkType())

	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())

	for packet := range packetSource.Packets() {
		// Extract Ethernet layer
		ethLayer := packet.Layer(layers.LayerTypeEthernet)
		if ethLayer == nil {
			continue
		}
		ethPacket, _ := ethLayer.(*layers.Ethernet)

		// Keep the original source MAC, change destination MAC to router's MAC
		ethPacket.DstMAC, _ = net.ParseMAC(routerMAC)

		// Extract IPv4 layer
		ipLayer := packet.Layer(layers.LayerTypeIPv4)
		if ipLayer == nil {
			continue
		}
		ipPacket, _ := ipLayer.(*layers.IPv4)

		// Modify Source & Destination IP
		ipPacket.SrcIP = net.ParseIP(newSourceIP)
		ipPacket.DstIP = net.ParseIP(newDestinationIP)
		ipPacket.Checksum = 0 // Reset checksum to be recalculated

		// Identify Transport Layer (TCP or UDP)
		var transportLayer gopacket.SerializableLayer
		var payload []byte

		if tcpLayer := packet.Layer(layers.LayerTypeTCP); tcpLayer != nil {
			tcpPacket, _ := tcpLayer.(*layers.TCP)
			tcpPacket.SetNetworkLayerForChecksum(ipPacket) // Ensure checksum is calculated correctly
			transportLayer = tcpPacket
			payload = tcpPacket.Payload
		} else if udpLayer := packet.Layer(layers.LayerTypeUDP); udpLayer != nil {
			udpPacket, _ := udpLayer.(*layers.UDP)
			udpPacket.SetNetworkLayerForChecksum(ipPacket) // Ensure checksum is calculated correctly
			transportLayer = udpPacket
			payload = udpPacket.Payload
		} else {
			continue // Skip non-TCP/UDP packets
		}

		// Serialize modified packet
		buf := gopacket.NewSerializeBuffer()
		opts := gopacket.SerializeOptions{
			FixLengths:       true, // Automatically adjust length fields
			ComputeChecksums: true, // Recalculate checksums
		}

		err := gopacket.SerializeLayers(buf, opts, ethPacket, ipPacket, transportLayer, gopacket.Payload(payload))
		if err != nil {
			log.Printf("Error serializing packet: %v", err)
			continue
		}

		// Write the modified packet
		err = writer.WritePacket(packet.Metadata().CaptureInfo, buf.Bytes())
		if err != nil {
			log.Printf("Error writing packet: %v", err)
		}
	}

	fmt.Printf("Rewritten pcap saved: %s\n", outputFile)
	return nil
}

func rewriteAllPcaps(inputDir, outputDir string) {
	files, err := filepath.Glob(filepath.Join(inputDir, "*.pcap"))
	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}

	for _, file := range files {
		outputFile := filepath.Join(outputDir, filepath.Base(file))
		err := rewritePcap(file, outputFile)
		if err != nil {
			log.Printf("Skipping file %s due to error: %v\n", file, err)
		}
	}
}

func main() {
	inputDir := "/home/umd-user/Downloads/archive/pcap_combined/Combined_Pcaps/Benign_Pcaps"
	outputDir := "./rewritten_pcaps"

	// Ensure output directory exists
	os.MkdirAll(outputDir, os.ModePerm)

	rewriteAllPcaps(inputDir, outputDir)
}

