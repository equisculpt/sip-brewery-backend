"""
Test Suite for Dual ZTECH API Integration
Tests the new dual ZTECH endpoints in the enhanced finance engine API
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, Any

class DualZTechAPITester:
    """Test suite for dual ZTECH API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_endpoint(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Test a specific API endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "endpoint": endpoint,
                        "response_code": response.status,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    return {
                        "status": "failed",
                        "endpoint": endpoint,
                        "response_code": response.status,
                        "error": error_text,
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            return {
                "status": "error",
                "endpoint": endpoint,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_comprehensive_tests(self):
        """Run comprehensive tests for all dual ZTECH endpoints"""
        print("🚀 Dual ZTECH API Comprehensive Test Suite")
        print("=" * 60)
        
        # Test endpoints
        test_cases = [
            {
                "name": "Natural Language Query - Z-Tech India",
                "endpoint": "/api/v2/ztech-dual/query",
                "params": {"query": "ztech india live price"}
            },
            {
                "name": "Natural Language Query - Zentech Systems",
                "endpoint": "/api/v2/ztech-dual/query",
                "params": {"query": "zentech systems comprehensive data"}
            },
            {
                "name": "Natural Language Query - Compare All",
                "endpoint": "/api/v2/ztech-dual/query",
                "params": {"query": "compare all ztech companies"}
            },
            {
                "name": "All Companies Data",
                "endpoint": "/api/v2/ztech-dual/all-companies",
                "params": None
            },
            {
                "name": "Z-Tech India Specific Data",
                "endpoint": "/api/v2/ztech-dual/ztech-india",
                "params": None
            },
            {
                "name": "Zentech Systems Specific Data",
                "endpoint": "/api/v2/ztech-dual/zentech-systems",
                "params": None
            },
            {
                "name": "Live Prices for All Companies",
                "endpoint": "/api/v2/ztech-dual/live-prices",
                "params": None
            },
            {
                "name": "Companies Comparison",
                "endpoint": "/api/v2/ztech-dual/comparison",
                "params": None
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"   Endpoint: {test_case['endpoint']}")
            print("-" * 50)
            
            result = await self.test_endpoint(
                test_case['endpoint'], 
                test_case['params']
            )
            
            results.append({
                "test_name": test_case['name'],
                "result": result
            })
            
            if result['status'] == 'success':
                print("✅ SUCCESS")
                
                # Extract key information from response
                data = result.get('data', {})
                
                if 'company_info' in data:
                    company_info = data['company_info']
                    print(f"   Company: {company_info.get('name', 'Unknown')}")
                    print(f"   Exchange: {company_info.get('exchange', 'Unknown')}")
                
                if 'live_data' in data and data['live_data']:
                    live_data = data['live_data']
                    print(f"   Price: ₹{live_data.get('current_price', 0)}")
                    print(f"   Volume: {live_data.get('volume', 0):,}")
                
                if 'ztech_india' in data and 'zentech_systems' in data:
                    print("   📊 Comparison Data Available")
                    
                if 'comparison' in data:
                    comparison = data['comparison']
                    print("   🔄 Comparison Analysis Generated")
                    
            elif result['status'] == 'failed':
                print("❌ FAILED")
                print(f"   Status Code: {result.get('response_code', 'Unknown')}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
            else:
                print("🚨 ERROR")
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Generate summary report
        await self.generate_summary_report(results)
        
        return results
    
    async def generate_summary_report(self, results: list):
        """Generate a summary report of all tests"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['result']['status'] == 'success')
        failed_tests = sum(1 for r in results if r['result']['status'] == 'failed')
        error_tests = sum(1 for r in results if r['result']['status'] == 'error')
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"🚨 Errors: {error_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\n🔍 DETAILED RESULTS:")
        print("-" * 40)
        
        for result in results:
            test_name = result['test_name']
            status = result['result']['status']
            
            if status == 'success':
                print(f"✅ {test_name}")
            elif status == 'failed':
                print(f"❌ {test_name} - Code: {result['result'].get('response_code', 'Unknown')}")
            else:
                print(f"🚨 {test_name} - Error: {result['result'].get('error', 'Unknown')}")
        
        print("\n🎯 RECOMMENDATIONS:")
        print("-" * 40)
        
        if successful_tests == total_tests:
            print("🏆 ALL TESTS PASSED! Dual ZTECH API is ready for production.")
        elif successful_tests >= total_tests * 0.8:
            print("🎉 Most tests passed. Minor issues need attention.")
        elif successful_tests >= total_tests * 0.5:
            print("⚠️  Some tests failed. Review and fix issues before deployment.")
        else:
            print("🚨 Major issues detected. Significant fixes required.")
        
        # Save detailed results to file
        report_data = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "error_tests": error_tests,
                "success_rate": (successful_tests/total_tests)*100
            },
            "detailed_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open("dual_ztech_api_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: dual_ztech_api_test_report.json")

async def test_dual_ztech_api():
    """Main test function"""
    async with DualZTechAPITester() as tester:
        await tester.run_comprehensive_tests()

if __name__ == "__main__":
    print("🔧 Starting Dual ZTECH API Test Suite...")
    print("⚠️  Note: Make sure the API server is running on http://localhost:8000")
    print("   Start with: python enhanced_finance_engine_api.py")
    print()
    
    try:
        asyncio.run(test_dual_ztech_api())
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n🚨 Test suite error: {e}")
