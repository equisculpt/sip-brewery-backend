const { AggregateRoot } = require('./base/AggregateRoot');
const { DomainEvent } = require('./base/DomainEvent');
const { Money } = require('./valueObjects/Money');
const { AssetAllocation } = require('./valueObjects/AssetAllocation');

module.exports = {
  AggregateRoot,
  DomainEvent,
  Money,
  AssetAllocation
};
