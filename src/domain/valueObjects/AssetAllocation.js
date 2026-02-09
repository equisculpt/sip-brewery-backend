/**
 * 📊 ASSET ALLOCATION VALUE OBJECT
 */
class AssetAllocation {
  constructor(equityPercentage, debtPercentage, hybridPercentage, otherPercentage) {
    this.equityPercentage = Number(equityPercentage);
    this.debtPercentage = Number(debtPercentage);
    this.hybridPercentage = Number(hybridPercentage);
    this.otherPercentage = Number(otherPercentage);
  }

  getTotalPercentage() {
    return Math.round(
      this.equityPercentage + this.debtPercentage + this.hybridPercentage + this.otherPercentage
    );
  }
}

module.exports = { AssetAllocation };
